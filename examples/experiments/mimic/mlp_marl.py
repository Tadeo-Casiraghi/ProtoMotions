# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.mimic.config import MimicEnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.agents.multi_agent_orchestrator.config import CoLearningConfig
import argparse


"""
Mimic Environment Configuration
================================

Full-body motion tracking environment with pose and velocity tracking.
Uses early termination on tracking error and bootstrapping at episode end.
"""


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot to add contact sensors for foot contact tracking."""
    # robot_cfg.update_fields(
    #     contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    # )
    pass

def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> MimicEnvConfig:
    """Build environment configuration (training defaults)."""
    from protomotions.envs.mimic.config import (
        MimicEarlyTerminationEntry,
        MimicObsConfig,
        MimicMotionManagerConfig,
    )
    from protomotions.envs.obs.config import FuturePoseType, MimicTargetPoseConfig
    from protomotions.envs.base_env.config import RewardComponentConfig
    from protomotions.envs.obs.config import HumanoidObsConfig, ActionHistoryConfig, MaxCoordsSelfObsConfig, ProstheticObsConfig
    from protomotions.envs.utils.rewards import (
        mean_squared_error_exp,
        rotation_error_exp,
        power_consumption_sum,
        norm,
        contact_mismatch_sum,
        impact_force_penalty,
        skin_pressure_penalty,
    )

    body_names = robot_cfg.kinematic_info.body_names
    all_dof_names = robot_cfg.kinematic_info.dof_names # This is UNRELIABLE for indices

    passive_dof_names = [
        "suspension_slide",
        "suspension_x",
        "suspension_y",
        "suspension_z",
        "R_Ankle_y",
    ]

    for i, name in enumerate(all_dof_names):
        if name == "R_Ankle_y":
            print(f"Found ankle DOF at index {i}")
            ankle_dof_index = i
        elif name == "Motor":
            print(f"Found motor DOF at index {i}")
            motor_dof_index = i
    
    for i, name in enumerate(body_names):
        if name == "prosthetic_assembly2":
            print(f"Found shank body at index {i}")
            shank_body_index = i
    
    
    # Store the defaults by NAME
    passive_defaults_by_name = {
        "suspension_slide": -0.025,
        "default": 0.0
    }


    mimic_early_termination = [
        MimicEarlyTerminationEntry(
            mimic_early_termination_key="max_joint_err",
            mimic_early_termination_thresh=0.5,
            less_than=False,
        )
    ]

    # Unified reward configuration - all components in one dict
    reward_config = {
        # Base rewards
        "action_smoothness": RewardComponentConfig(
            function=norm,
            variables={
                "x": "current_actions - previous_actions",
            },
            weight=-0.08,
        ),
        # Mimic tracking rewards
        "gt_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_pos",
                "ref_x": "ref_state.rigid_body_pos",
                "coefficient": "-100.0",
            },
            indices_subset=["tracking_bodies"],
            weight=0.4,
        ),
        "skin_rew": RewardComponentConfig(
            function=skin_pressure_penalty,
            variables={
                "contact_forces": "current_state.rigid_body_contact_forces",
                "body_quats": "current_state.rigid_body_rot",
            },
            indices_subset=["skin_bodies"],
            weight=-1e-4,  # Negative = Penalty
            min_value=-0.5,
        ),
        "gr_rew": RewardComponentConfig(
            function=rotation_error_exp,
            variables={
                "q": "current_state.rigid_body_rot",
                "ref_q": "ref_state.rigid_body_rot",
                "coefficient": "-5.0",
            },
            indices_subset=["tracking_bodies"],
            weight=0.3,
        ),
        "gv_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_vel",
                "ref_x": "ref_state.rigid_body_vel",
                "coefficient": "-0.5",
            },
            indices_subset=["tracking_bodies"],
            weight=0.1,
        ),
        "gav_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_ang_vel",
                "ref_x": "ref_state.rigid_body_ang_vel",
                "coefficient": "-0.1",
            },
            indices_subset=["tracking_bodies"],
            weight=0.1,
        ),
        "rh_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_pos[:, 0, 2]",  # Root height (z-coord of body 0)
                "ref_x": "ref_state.rigid_body_pos[:, 0, 2]",
                "coefficient": "-100.0",
            },
            weight=0.1,
        ),
        "pow_rew": RewardComponentConfig(
            function=power_consumption_sum,
            variables={
                "dof_forces": "current_state.dof_forces",
                "dof_vel": "current_state.dof_vel",
                "use_torque_squared": "False",
                "indices": "humanoid_joints",  # Only penalize power for humanoid joints, not prosthetic
            },
            weight=-7.5e-5,
            min_value=-0.75,
            zero_during_grace_period=False,
            # TADEO ACA HAY QUE REVISAR ESTO indices_subset=["all_physical_dofs"]
        ),
        "contact_match_rew": RewardComponentConfig(
            function=contact_mismatch_sum,
            variables={
                "sim_contacts": "current_state.rigid_body_contacts",
                "ref_contacts": "ref_state.rigid_body_contacts",
            },
            indices_subset=["all_left_foot_bodies"], #, "all_right_foot_bodies"],
            weight=-0.2,
            zero_during_grace_period=True,
        ),
        "contact_force_change_rew": RewardComponentConfig(
            function=impact_force_penalty,
            variables={
                "current_forces": "current_contact_force_magnitudes",
                "previous_forces": "prev_contact_force_magnitudes",
            },
            indices_subset=["all_left_foot_bodies"], #, "all_right_foot_bodies"],
            weight=-1e-5,
            min_value=-0.5,
            zero_during_grace_period=True,
        ),
    }

    secondary_reward_config = {
        "action_smoothness": RewardComponentConfig(
            function=norm,
            variables={"x": "current_actions - previous_actions"},
            weight=-0.05,  # Maybe stiffer penalty for prosthetic?
        ),
        "skin_rew": RewardComponentConfig(
            function=skin_pressure_penalty,
            variables={
                "contact_forces": "current_state.rigid_body_contact_forces",
                "body_quats": "current_state.rigid_body_rot",
            },
            indices_subset=["skin_bodies"],
            weight=-1e-3, 
            min_value=-1.0,
        ),
    }




    env_config: MimicEnvConfig = MimicEnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        humanoid_obs=HumanoidObsConfig(
            max_coords_obs=MaxCoordsSelfObsConfig(
                enabled=True,
                observe_contacts=True,
            ),
            action_history=ActionHistoryConfig(
                enabled=True,
                num_historical_steps=1,
            ),
        ),
        prosthetic_obs=ProstheticObsConfig(
            enabled = True,
            num_historical_steps = 5,
            ankle_dof_index = ankle_dof_index,
            motor_dof_index = motor_dof_index,
            shank_body_index = shank_body_index,
        ),
        reward_config=reward_config,
        mimic_early_termination=mimic_early_termination,
        mimic_bootstrap_on_episode_end=True,
        mimic_obs=MimicObsConfig(
            enabled=True,
            mimic_target_pose=MimicTargetPoseConfig(
                enabled=True, type=FuturePoseType.MAX_COORDS, with_velocities=True
            ),
        ),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
        passive_dof_names=passive_dof_names,
        passive_defaults_map=passive_defaults_by_name,
        active_dof_indices=None,
        passive_dof_defaults=None,

        secondary_reward_flag=True,
        secondary_reward_config=secondary_reward_config,
    )
    return env_config


def humanoid_agent_config(
    robot_config: RobotConfig, env_config: MimicEnvConfig, args: argparse.Namespace, agent_type: str
) -> PPOAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

    body_names = robot_config.kinematic_info.body_names
    body_indices_to_remove = []
    contact_indices_to_remove = []

    for i, name in enumerate(body_names):
        n_low = name.lower()
        if (
            "prosthetic" in n_low 
            or "skin" in n_low 
            or "socket" in n_low 
            or name in ["R_Ankle", "R_Toe"] # Ensure these match your URDF exact casing
        ):
            body_indices_to_remove.append(i)
        if (name in ["R_Ankle", "R_Toe"]):
            contact_indices_to_remove.append(i)
    
    dofs = robot_config.kinematic_info.dof_names
    action_indices = []

    for i, dof_name in enumerate(dofs):
        n_low = dof_name.lower()
        if (
            "suspension" in n_low 
            or dof_name in ["R_Ankle_y", "Motor"] # Ensure these match your URDF exact casing
        ):
            print(dof_name, " Skipped")
            continue  # Skip this DOF
        print(dof_name, " Included")
        action_indices.append(i)

    num_active_actions = len(action_indices)
    print("Number of humanoid active joints:", num_active_actions)

    actor_config = PPOActorConfig(
        num_out=num_active_actions,
        actor_logstd=-2.9,
        in_keys=["blind_body_obs", "mimic_target_poses", "agent_action_history"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "blind_body_obs",
                "mimic_target_poses",
                "agent_action_history",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=num_active_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
            output_activation="tanh",
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["blind_body_obs", "mimic_target_poses", "agent_action_history"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )
    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                "blind_body_obs",
                "mimic_target_poses",
                "agent_action_history",
            ],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,

        use_blind_body_indices=True,
        body_indices_to_remove=body_indices_to_remove,
        contact_indices_to_remove=contact_indices_to_remove,
        total_num_bodies=len(body_names),

        evaluator=MimicEvaluatorConfig(
            eval_metric_keys=[
                "gt_err",
                "gr_err",
                "gr_err_degrees",
                "lr_err_degrees",
                "gt_rew",
                "gr_rew",
                "pow_rew",
                "contact_force_change_rew",
            ],
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True
        ),
        action_indices=action_indices,
    )

    print(f"[{agent_type.upper()}] Config: Controlling {num_active_actions} DOFs")
    return agent_config

def prosthetic_agent_config(
    robot_config: RobotConfig, env_config: MimicEnvConfig, args: argparse.Namespace, agent_type: str
) -> PPOAgentConfig:
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig

    gSDE = False  # Enable gSDE for the prosthetic agent
    dofs = robot_config.kinematic_info.dof_names
    action_indices = []

    for i, dof_name in enumerate(dofs):
        if dof_name == "Motor":
            action_indices.append(i)

    actor_config = PPOActorConfig(
        num_out=3,
        actor_logstd=-2.9,
        in_keys=["prosthetic_obs", "prosthetic_previous_actions", "historical_prosthetic_obs"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "prosthetic_obs",
                "prosthetic_previous_actions",
                "historical_prosthetic_obs",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=3,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
            output_activation="tanh",
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["prosthetic_obs", "prosthetic_previous_actions", "historical_prosthetic_obs"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )
    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                "prosthetic_obs",
                "prosthetic_previous_actions",
                "historical_prosthetic_obs",
            ],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,

        use_blind_body_indices=True,

        save_actions = True,
        action_history_length = 5,

        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True
        ),
        action_indices=action_indices,
        gSDE=gSDE,  # Pass the gSDE flag to the agent config
    )
    return agent_config

def agent_config(
    robot_config: RobotConfig, env_config: MimicEnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:


    humanoid_agent_cfg = humanoid_agent_config(robot_config, env_config, args, agent_type="humanoid")
    prosthetic_agent_cfg = prosthetic_agent_config(robot_config, env_config, args, agent_type="prosthetic")

    agent_config = CoLearningConfig(
        agents={
            "humanoid": humanoid_agent_cfg,
            "prosthetic": prosthetic_agent_cfg,
        },
        sync_updates=True,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
    )

    return agent_config


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    # For mimic: disable early termination during evaluation
    if env_cfg is not None:
        if hasattr(env_cfg, "mimic_early_termination"):
            env_cfg.mimic_early_termination = None
        if hasattr(env_cfg, "max_episode_length"):
            env_cfg.max_episode_length = 1000000
        if hasattr(env_cfg, "motion_manager"):
            if hasattr(env_cfg.motion_manager, "resample_on_reset"):
                env_cfg.motion_manager.resample_on_reset = True
            if hasattr(env_cfg.motion_manager, "init_start_prob"):
                env_cfg.motion_manager.init_start_prob = 1.0


