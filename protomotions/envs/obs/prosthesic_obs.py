import torch
from torch import Tensor
from protomotions.envs.obs.config import ProstheticObsConfig
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.envs.utils.math import quat_rotate_inverse

# Import HistoryBuffer. If it's in humanoid.py, you might need to move it 
# to a common utils file or import it from there. 
# For now, assuming you can import it or paste the class definition here.
from protomotions.envs.obs.humanoid import HistoryBuffer 

def compute_prosthetic_observations(
    dof_pos: Tensor,
    body_rot: Tensor,
    body_ang_vel: Tensor,
    ankle_dof_idx: int,
    shank_body_idx: int,
    w_last: bool = True,
) -> Tensor:
    """
    Computes: [Ankle Angle (1) | Local Gyro (3) | Local Gravity (3)]
    Total size: 7 floats per environment
    """
    # 1. Ankle Angle (1 dim)
    # Assumes 1-DOF ankle. 
    ankle_angle = dof_pos[:, ankle_dof_idx].unsqueeze(-1)

    # TODO: Check all of this and maybe add noise, also add torque and motor angle
    # 2. Simulated Gyro: Local Angular Velocity
    # Rotate global angular vel into the shank's local frame
    global_ang_vel = body_ang_vel[:, shank_body_idx]
    shank_rot = body_rot[:, shank_body_idx]
    local_ang_vel = quat_rotate_inverse(shank_rot, global_ang_vel, w_last=w_last)

    # 3. Simulated Accelerometer: Projected Gravity
    # Gravity vector [0, 0, -1] in global frame, rotated to local frame
    # (The sensor feels "up" acceleration equal to 1g)
    gravity_vec = torch.zeros_like(global_ang_vel)
    gravity_vec[:, 2] = -1.0 
    local_gravity = quat_rotate_inverse(shank_rot, gravity_vec, w_last=w_last)

    return torch.cat([ankle_angle, local_ang_vel, local_gravity], dim=-1)


class ProstheticObs:
    """
    Handles computation of prosthetic leg observations (IMU + Joint Angle).
    Maintains a history buffer similar to HumanoidObs.
    """

    def __init__(self, config: ProstheticObsConfig, env):
        self.config = config
        self.env = env
        self.device = self.env.device

        # Buffers
        self.prosthetic_obs = None
        self.prosthetic_obs_hist_buf = None
        self._initialized = False

    def post_physics_step(self):
        """Called every simulation step to rotate buffers and compute new obs."""
        if not self._initialized:
            self.compute_observations(torch.arange(self.env.num_envs, device=self.device))

        # Rotate history buffer (discard oldest, make room for new)
        if self.prosthetic_obs_hist_buf is not None:
            self.prosthetic_obs_hist_buf.rotate()

    def reset_hist(self, env_ids):
        """Reset history for specific environments (e.g. after a fall)."""
        if not self._initialized:
            self.compute_observations(env_ids)

        if self.config.num_historical_steps > 1:
            # Fill the entire history with the current frame to avoid "ghost" data
            current_obs = self.prosthetic_obs_hist_buf.get_current(env_ids)
            
            # Repeat current frame across all history steps
            # Shape: [num_hist, num_envs_reset, obs_dim]
            filled_hist = current_obs.unsqueeze(0).repeat(
                self.config.num_historical_steps, 1, 1
            )
            self.prosthetic_obs_hist_buf.set_all(filled_hist, env_ids=env_ids)

    def compute_observations(self, env_ids):
        """Actual math computation for the current frame."""
        self._initialized = True

        # 1. Fetch State from Simulator
        current_state: RobotState = self.env.simulator.get_robot_state(env_ids)

        # 2. Compute Math
        obs = compute_prosthetic_observations(
            dof_pos=current_state.dof_pos,
            body_rot=current_state.rigid_body_rot,
            body_ang_vel=current_state.rigid_body_ang_vel,
            ankle_dof_idx=self.config.ankle_dof_index,
            shank_body_idx=self.config.shank_body_index,
            w_last=True 
        )

        # 3. Initialize Buffers (First Run Only)
        if self.prosthetic_obs is None:
            self.prosthetic_obs = torch.zeros(
                self.env.num_envs,
                obs.shape[-1],
                dtype=torch.float,
                device=self.device,
            )
            self.prosthetic_obs_hist_buf = HistoryBuffer(
                self.config.num_historical_steps,
                self.env.num_envs,
                shape=(obs.shape[-1],),
                device=self.device,
            )

        # 4. Store Data
        self.prosthetic_obs[env_ids] = obs
        self.prosthetic_obs_hist_buf.set_curr(obs, env_ids)

    def get_obs(self):
        """Return the dictionary of observations to be fed to the neural net."""
        if not self._initialized:
            self.compute_observations(torch.arange(self.env.num_envs, device=self.device))
        
        obs = {}
        
        # 1. Current Frame (Optional, usually Policy wants history)
        # obs["prosthetic_obs"] = self.prosthetic_obs.clone()
        
        # 2. Flattened History (Current + Past)
        # This creates a vector like: [t=0, t=-1, t=-2, ...]
        obs["historical_prosthetic_obs"] = (
            self.prosthetic_obs_hist_buf.get_all_flattened().clone()
        )

        return obs