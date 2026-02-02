from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.mimic.config import MimicEnvConfig
from protomotions.agents.multi_agent_orchestrator.config import CoLearningConfig
from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.ppo.config import PPOAgentConfig, PPOModelConfig, PPOActorConfig, AdvantageNormalizationConfig
from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.envs.base_env.config import RewardComponentConfig
from protomotions.envs.utils.rewards import norm, skin_pressure_penalty

import argparse
import copy

# 1. INHERIT: Import everything from your base experiment
#    (Assumes mlp.py is in the same folder or python path)
try:
    from .mlp import (
        configure_robot_and_simulator,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        apply_inference_overrides,
        env_config as base_env_config  # Rename to avoid conflict
    )
except ImportError:
    # Fallback if running as script
    from mlp import (
        configure_robot_and_simulator,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        apply_inference_overrides,
        env_config as base_env_config
    )

# 2. OVERRIDE: Environment Configuration
def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> MimicEnvConfig:
    """
    Inherits the base environment but adds the Secondary Reward Config
    for the prosthetic agent.
    """
    # A. Get the standard configuration
    cfg = base_env_config(robot_cfg, args)

    # B. Activate Secondary Rewards
    cfg.secondary_reward_flag = True

    # C. Define Prosthetic-Specific Rewards
    #    (This dict will be used by the "leg" agent)
    cfg.secondary_reward_config = {
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
                "indices" : "[ 7,  8,  9, 10, 11, 12, 13, 14]",
            },
            weight=-1e-3, 
            min_value=-1.0,
        ),
    }

    # # D. (Optional) Modify the Base (Humanoid) Rewards
    # #    Example: Remove skin reward from the main body if it only applies to the leg
    # if "skin_rew" in cfg.reward_config:
    #     del cfg.reward_config["skin_rew"]

    return cfg


# 3. OVERRIDE: Agent Configuration (The Orchestrator)
def agent_config(
    robot_config: RobotConfig, env_config: MimicEnvConfig, args: argparse.Namespace
) -> CoLearningConfig:
    
    # --- Define Split ---
    total_dofs = robot_config.kinematic_info.num_dofs
    LEG_DOFS = 2  # Update this to match your actual prosthetic joints
    BODY_DOFS = total_dofs - LEG_DOFS

    mapping_info = {
        "humanoid": {
            "obs": [0, 1000],        
            "act": [0, BODY_DOFS]    
        },
        "leg": {
            "obs": [0, 1000],             
            "act": [BODY_DOFS, total_dofs] 
        }
    }

    def create_ppo_config(agent_name, num_output_actions):
        
        # Network config (Same architecture as mlp.py)
        actor_config = PPOActorConfig(
            num_out=num_output_actions, # <--- Specific to this agent
            actor_logstd=-2.9,
            in_keys=["max_coords_obs", "mimic_target_poses", "historical_previous_actions"],
            mu_key="actor_trunk_out",
            mu_model=MLPWithConcatConfig(
                in_keys=[
                    "max_coords_obs",
                    "mimic_target_poses",
                    "historical_previous_actions",
                ],
                normalize_obs=True,
                norm_clamp_value=5,
                out_keys=["actor_trunk_out"],
                num_out=num_output_actions, # <--- Specific to this agent
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
                output_activation="tanh",
            ),
        )

        critic_config = MLPWithConcatConfig(
            in_keys=["max_coords_obs", "mimic_target_poses", "historical_previous_actions"],
            out_keys=["value"],
            normalize_obs=True,
            norm_clamp_value=5,
            num_out=1,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
        )

        return PPOAgentConfig(
            model=PPOModelConfig(
                in_keys=[
                    "max_coords_obs",
                    "mimic_target_poses",
                    "historical_previous_actions",
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
            # Evaluator runs per-agent metrics
            evaluator=MimicEvaluatorConfig(
                eval_metric_keys=[
                    "gt_err", "gr_rew", "pow_rew"
                ],
            ),
            advantage_normalization=AdvantageNormalizationConfig(
                enabled=True, shift_mean=True
            ),
        )

    # -----------------------------------------------------------
    # C. Return the Co-Learning Config
    # -----------------------------------------------------------
    return CoLearningConfig(
        # Global Settings
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_steps=32,
        
        # Orchestrator Settings
        mapping_info=mapping_info,
        agents={
            "humanoid": create_ppo_config("humanoid", BODY_DOFS),
            "leg":      create_ppo_config("leg", LEG_DOFS)
        }
    )