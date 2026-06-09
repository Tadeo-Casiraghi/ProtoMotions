import torch
from torch import Tensor, nn
from protomotions.envs.obs.config import ProstheticObsConfig
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.utils import rotations

class HistoryBuffer(nn.Module):
    """Circular buffer for storing temporal history of observations or actions.

    Stores the past N frames of data where index 0 is the most recent frame.
    Used for temporal observations like action history or historical poses.

    Args:
        num_steps: Number of historical timesteps to store.
        num_envs: Number of parallel environments.
        shape: Shape of each data element (default: scalar).
        dtype: Data type for storage.
        device: Device for tensor storage.
    """

    data: Tensor

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        shape: tuple = (),
        dtype=torch.float,
        device="cpu",
    ):
        super().__init__()
        data = torch.zeros(num_steps, num_envs, *shape, dtype=dtype, device=device)
        self.register_buffer("data", data, persistent=False)
        self.to(device)

    def rotate(self):
        """Shift history forward by one timestep (oldest frame is discarded)."""
        self.data = self.data.roll(
            shifts=1, dims=0
        )  # equivalent to self.data[i + 1] = self.data[i]

    @torch.no_grad()
    def update(self, fresh_data: Tensor):
        """Rotate buffer and update current frame with new data.

        Args:
            fresh_data: New data to insert at current frame [num_envs, *shape]
        """
        self.rotate()
        self.set_curr(fresh_data)

    @torch.no_grad()
    def set_all(self, fresh_data: Tensor, env_ids=slice(None)):
        """Set all timesteps for specified environments.

        Args:
            fresh_data: Data for all timesteps [num_steps, num_envs, *shape]
            env_ids: Environment indices to update
        """
        self.data[:, env_ids] = fresh_data

    @torch.no_grad()
    def set_hist(self, fresh_data: Tensor, env_ids=slice(None)):
        """Set historical data (excluding current frame).

        Args:
            fresh_data: Historical data [num_steps-1, num_envs, *shape]
            env_ids: Environment indices to update
        """
        self.data[1:, env_ids] = fresh_data

    @torch.no_grad()
    def set_curr(self, fresh_data: Tensor, env_ids=slice(None)):
        """Set current frame data.

        Args:
            fresh_data: Current frame data [num_envs, *shape]
            env_ids: Environment indices to update
        """
        self.data[0, env_ids] = fresh_data

    def get_hist(self, env_ids=slice(None)):
        """Get historical data (excluding current frame).

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            Historical data [num_steps-1, num_envs, *shape]
        """
        return self.data[1:, env_ids]

    def get_current(self, env_ids=slice(None)):
        """Get current frame data.

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            Current frame data [num_envs, *shape]
        """
        return self.data[0, env_ids]

    def get_all(self, env_ids=slice(None)):
        """Get all timesteps.

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            All historical data [num_steps, num_envs, *shape]
        """
        return self.data[:, env_ids]

    def get_all_flattened(self, env_ids=slice(None)):
        """Get all timesteps flattened into a single feature vector per environment.

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            Flattened history [num_envs, num_steps * features]
        """
        data = self.get_all(env_ids)
        num_envs = data.shape[1]
        return data.permute(1, 0, 2).reshape(num_envs, -1)

    def get_index(self, idx: int, env_ids=slice(None)):
        """Get data at specific timestep index.

        Args:
            idx: Timestep index (0 = current, 1 = previous, etc.)
            env_ids: Environment indices to retrieve

        Returns:
            Data at specified timestep [num_envs, *shape]
        """
        return self.data[idx, env_ids]

    @property
    def device(self) -> torch.device:
        """Get device from registered buffers."""
        return self.data.device 

@torch.jit.script
def compute_prosthetic_observations(
    ankle_dof_index: int,
    motor_dof_index: int,
    shank_body_index: int,
    dof_pos: Tensor,
    dof_vel: Tensor,
    body_lin_acc: Tensor,
    body_rot: Tensor,
    body_ang_vel: Tensor,
    debug_step: int = 0,
    w_last: bool = True,
) -> Tensor:
        """
        Computes: 
            1.  Ankle Angle (from dof_pos)
            2.  Ankle Velocity (from dof_pos history)
            3.  Ankle Motor Angle (from dof_pos)
            4.  Ankle Motor Velocity (from dof_pos history)
            9.  Simulated Gyro: Local Angular Velocity (from body_ang_vel + body_rot)
            10.  Simulated Accelerometer: Projected Gravity (from body_rot)
        """
        debug = False

        step = debug_step  # For easier debugging and visualization in tensorboard

        # Convert step to a float offset: Step 1 = 0.001, Step 2 = 0.002, etc.
        time_offset = float(step) * 0.001

        # 1. Ankle Angle (1 dim)
        ankle_angle = dof_pos[:, ankle_dof_index].unsqueeze(-1)
        if debug:
            # turn to 1.0 for debugging to see if this value is seen in the prosthetic input
            ankle_angle = torch.ones_like(ankle_angle) * (1.0 + time_offset)

        # 2. Ankle Velocity (1 dim)
        ankle_vel = dof_vel[:, ankle_dof_index].unsqueeze(-1)
        if debug:
            # turn to 2.0 for debugging to see if this value is seen in the prosthetic input
            ankle_vel = torch.ones_like(ankle_vel) * (2.0 + time_offset)
        
        # 3. Ankle Motor Angle (1 dim)
        motor_angle = dof_pos[:, motor_dof_index].unsqueeze(-1)
        if debug:
            # turn to 3.0 for debugging to see if this value is seen in the prosthetic input
            motor_angle = torch.ones_like(motor_angle) * (3.0 + time_offset)

        # 4. Ankle Motor Velocity (1 dim)
        motor_vel = dof_vel[:, motor_dof_index].unsqueeze(-1)
        if debug:
            # turn to 4.0 for debugging to see if this value is seen in the prosthetic input
            motor_vel = torch.ones_like(motor_vel) * (4.0 + time_offset)

        # --- Simulated IMU ---

        shank_rot = body_rot[:, shank_body_index]

        # 9. Gyro: Local Angular Velocity
        global_ang_vel = body_ang_vel[:, shank_body_index]
        rot_inv = rotations.quat_conjugate(shank_rot, w_last)
        local_ang_vel = rotations.quat_rotate(rot_inv, global_ang_vel, w_last)
        if debug:
            # turn to 5.0, 6.0, 7.0 for debugging to see if these values are seen in the prosthetic input
            base_gyro = torch.tensor([5.0, 6.0, 7.0], device=local_ang_vel.device)
            local_ang_vel = torch.ones_like(local_ang_vel) * (base_gyro + time_offset)

        # 10. Accelerometer: Proper Acceleration
    
        # A. Get exact kinematic acceleration from simulator
        global_lin_acc = body_lin_acc[:, shank_body_index]
        
        # B. Add Gravity Component to get Proper Acceleration
        # IMU reads 1g (9.81) upwards when stationary.
        gravity_vec = torch.zeros_like(global_lin_acc)
        gravity_vec[:, 2] = 9.81
        
        global_proper_acc = global_lin_acc + gravity_vec
        
        # C. Rotate to Local Frame
        local_acc = rotations.quat_rotate(rot_inv, global_proper_acc, w_last)
        if debug:
            # turn to 8.0, 9.0, 10.0 for debugging to see if these values are seen in the prosthetic input
            base_acc = torch.tensor([8.0, 9.0, 10.0], device=local_acc.device)
            local_acc = torch.ones_like(local_acc) * (base_acc + time_offset)

        # --- Combine ---
        return torch.cat([
            ankle_angle, 
            ankle_vel, 
            motor_angle, 
            motor_vel, 
            local_ang_vel, 
            local_acc
        ], dim=-1)


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

        self.previous_actions_hist_buf = None
        if self.config.action_history.enabled:
            self.previous_actions_hist_buf = HistoryBuffer(
                self.config.action_history.num_historical_steps,
                self.env.num_envs,
                shape=(3,),
                device=self.env.device,
            )

        self._initialized = False

        self.step = 0  # For debugging purposes to track time in compute_observations

    def post_physics_step(self):
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        
        if not self._initialized:
            self.compute_observations(env_ids)
            return  # buffers just initialized, no need to rotate

        # Rotate first (push current → history), then compute new current
        if self.prosthetic_obs_hist_buf is not None:
            self.prosthetic_obs_hist_buf.rotate()

        if self.config.action_history.enabled:
            self.previous_actions_hist_buf.rotate()
        
        self.compute_observations(env_ids)

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

        if self.config.action_history.enabled and self.config.action_history.num_historical_steps > 1:
            current_actions = self.previous_actions_hist_buf.get_current(env_ids)
            filled_actions_hist = current_actions.unsqueeze(0).repeat(
                self.config.action_history.num_historical_steps, 1, 1
            )
            self.previous_actions_hist_buf.set_all(filled_actions_hist, env_ids=env_ids)

    def compute_observations(self, env_ids):
        """Actual math computation for the current frame."""
        self._initialized = True

        # 1. Fetch State from Simulator
        current_state: RobotState = self.env.simulator.get_robot_state(env_ids)

        # 2. Compute Math
        obs = compute_prosthetic_observations(
                ankle_dof_index=self.config.ankle_dof_index,
                motor_dof_index=self.config.motor_dof_index,
                shank_body_index=self.config.shank_body_index,
                dof_pos=current_state.dof_pos,
                dof_vel=current_state.dof_vel,
                body_lin_acc=current_state.rigid_body_acc,
                body_rot=current_state.rigid_body_rot,
                body_ang_vel=current_state.rigid_body_ang_vel,
                w_last=True,
        )

        sim_ref = self.env.simulator

        # Pull the tensor for the specific envs and joint index
        # Shape goes from [num_envs, num_dofs] -> [len(env_ids)] -> [len(env_ids), 1]
        if hasattr(sim_ref, "_robot") and hasattr(sim_ref._robot, "data") and hasattr(sim_ref._robot.data, "applied_torque"):
            prosthetic_torque = sim_ref._robot.data.applied_torque[env_ids, self.config.motor_dof_index].unsqueeze(-1)
        else:
            # Fallback if the cache isn't populated yet during the very first cold-start frame
            prosthetic_torque = torch.zeros(len(env_ids), 1, dtype=torch.float, device=self.device)

        # A quick pre-scale so 100Nm becomes 1.0, keeping it well under your clamp of 5
        prosthetic_torque = prosthetic_torque * 0.04

        obs = torch.cat([obs, prosthetic_torque], dim=-1)

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

            if self.config.action_history.enabled:
                all_prev_actions = self.env.simulator.get_previous_actions(env_ids)
                
                # Pull theta target from its physical joint slot
                physical_action = all_prev_actions[:, self.config.motor_dof_index].unsqueeze(-1)
                
                # Pull Kp and Kd from the final 2 tail slots of the tensor
                extra_actions = all_prev_actions[:, -2:] 
                
                # Stitch them back into the true [theta, Kp, Kd] format the network expects
                prosthetic_actions = torch.cat([physical_action, extra_actions], dim=-1)
                
                self.previous_actions_hist_buf.set_curr(prosthetic_actions, env_ids)

        # 4. Store Data
        self.prosthetic_obs[env_ids] = obs
        self.prosthetic_obs_hist_buf.set_curr(obs, env_ids)

        if self.config.action_history.enabled:
            all_prev_actions = self.env.simulator.get_previous_actions(env_ids)
                
            # Pull theta target from its physical joint slot
            physical_action = all_prev_actions[:, self.config.motor_dof_index].unsqueeze(-1)
            
            # Pull Kp and Kd from the final 2 tail slots of the tensor
            extra_actions = all_prev_actions[:, -2:] 
            
            # Stitch them back into the true [theta, Kp, Kd] format the network expects
            prosthetic_actions = torch.cat([physical_action, extra_actions], dim=-1)

            self.previous_actions_hist_buf.set_curr(prosthetic_actions, env_ids)

    def get_obs(self):
        """Return the dictionary of observations to be fed to the neural net."""
        if not self._initialized:
            self.compute_observations(torch.arange(self.env.num_envs, device=self.device))
        
        obs = {}
        
        # 1. Current Frame (Optional, usually Policy wants history)
        obs["prosthetic_obs"] = self.prosthetic_obs.clone()
        
        # 2. Flattened History (Current + Past)
        # This creates a vector like: [t=0, t=-1, t=-2, ...]
        obs["historical_prosthetic_obs"] = (
            self.prosthetic_obs_hist_buf.get_all_flattened().clone()
        )

        if self.config.action_history.enabled:
            obs["historical_prosthetic_previous_actions"] = (
                self.previous_actions_hist_buf.get_all_flattened().clone()
            )

        return obs