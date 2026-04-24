import torch
from torch import Tensor
import time
import logging
from typing import Dict, Any
from rich.progress import track

from pathlib import Path
from protomotions.agents.utils.data import ExperienceBuffer
from protomotions.agents.utils.metering import TimeReport
from protomotions.agents.utils.training import aggregate_scalar_metrics

from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator

from protomotions.agents.ppo.agent import PPO 

log = logging.getLogger(__name__)

@torch.jit.script
def compute_torques(
    kp: Tensor,
    kd: Tensor,
    desired_angle: Tensor,
    current_angle: Tensor,
    current_vel: Tensor
) -> Tensor:

    
    torque = kp * (desired_angle - current_angle) - kd * current_vel
    
    # Optional: Clamp torque to motor limits if needed
    # torque = torch.clamp(torque, -50.0, 50.0)
    
    return torque

class CoLearningMimicEvaluator(MimicEvaluator):
    """
    A wrapper around MimicEvaluator that knows how to step TWO agents
    simultaneously during the evaluation loop.
    """
    def __init__(self, humanoid_agent, prosthetic_agent, fabric, config):
        # Initialize the base MimicEvaluator with the Humanoid (primary)
        super().__init__(humanoid_agent, fabric, config)
        self.prosthetic_agent = prosthetic_agent

    def evaluate_episode(
        self,
        metrics: dict,
        active_env_ids: torch.Tensor,
        active_motion_ids: torch.Tensor,
    ) -> None:
        """
        Overrides the single-agent evaluation loop to support Co-Learning.
        """
        assert len(active_env_ids) == len(active_motion_ids)

        # 1. Setup Motion Manager (Same as original)
        self.motion_manager.motion_ids[active_env_ids] = active_motion_ids
        max_len = self.motion_lib.get_motion_num_frames(active_motion_ids).max().item()
        max_len = min(max_len, self.config.max_eval_steps)
        self.motion_manager.motion_times[active_env_ids] = 0.0

        # 2. Reset Environment (Same as original)
        obs, _ = self.env.reset(
            active_env_ids, sample_flat=True, disable_motion_resample=True
        )
        
        # --- MULTI-AGENT OBSERVATION CHAIN ---
        obs = self.agent.add_agent_info_to_obs(obs)                  # Humanoid
        obs = self.prosthetic_agent.add_agent_history_to_obs(obs)    # Prosthetic
        obs_td = self.agent.obs_dict_to_tensordict(obs)
        
        # Zero out prosthetic history for these envs
        if self.prosthetic_agent.save_actions:
            # Create a zero tensor matching the action shape from the buffer
            zero_action = torch.zeros(
                self.prosthetic_agent.num_envs, 
                self.prosthetic_agent.action_history.data.shape[-1], 
                device=self.prosthetic_agent.device
            )
            self.prosthetic_agent.action_history.set_all(zero_action)
            
            if hasattr(self.prosthetic_agent, 'prev_command_buffer'):
                self.prosthetic_agent.prev_command_buffer.zero_()

        # 3. Evaluation Loop
        for step in range(max_len):
            
            # --- A. Humanoid Action (Deterministic) ---
            # We prefer 'mean_action' for evaluation if available
            h_outs = self.agent.model(obs_td)
            action_h = h_outs.get("mean_action", h_outs.get("action"))
            
            # --- B. Prosthetic Action (Deterministic + Torque Calc) ---
            # We need to manually do the 'collect_rollout_step_withcalc' logic 
            # but forcing Deterministic (Mean) actions.
            
            p_outs = self.prosthetic_agent.model(obs_td)
            
            # 1. Get Mean Raw Actions
            raw_action_p = p_outs.get("mean_action", p_outs.get("action"))
            
            # 2. Re-implement torque calculation (Deterministic)
            # (Copying the logic from your collect_rollout_step_withcalc)
            # NOTE: We can't call collect_rollout_step_withcalc directly because 
            # it might sample stochastic actions.
            
            # Extract raw params
            raw_theta = raw_action_p[:, 0]
            raw_kp    = raw_action_p[:, 1]
            raw_kd    = raw_action_p[:, 2]

            kp_scale = 1000.0
            kp_offset = 1000.0 
            
            # Example: Map [-1, 1] -> [0, 10] for Kd
            kd_scale = 5
            kd_offset = 5
            
            # Example: Map [-1, 1] -> [-pi, pi] for Angle
            angle_scale = 3.14
            angle_offset = 0.0

            desired_angle = raw_theta * angle_scale + angle_offset
            kp_phys       = raw_kp * kp_scale + kp_offset
            kd_phys       = raw_kd * kd_scale + kd_offset

            # Get State
            current_angle = obs_td["prosthetic_obs"][:, 0]
            current_vel   = obs_td["prosthetic_obs"][:, 1]
            
            # JIT Torque Calc
            # We need to import the JIT function or access it from the agent module
            # Assuming compute_torques_jit is available here
            torque = compute_torques(
                desired_angle, kp_phys, kd_phys, current_angle, current_vel
            )
            
            # Construct Final Prosthetic Action
            action_p = torque.unsqueeze(-1)
            
            # --- C. Merge Actions ---
            env_action = self.agent.expand_action_to_env(action_h)
            env_action += self.prosthetic_agent.expand_action_to_env(action_p)

            # --- D. Step Environment ---
            obs, rewards, dones, terminated, extras = self.env.step(env_action)
            
            # --- E. Prepare Next Step ---
            obs = self.agent.add_agent_info_to_obs(obs)
            obs = self.prosthetic_agent.add_agent_history_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)
            
            # Update Prosthetic History (with the deterministic actions)
            if self.prosthetic_agent.save_actions:
                 combined_entry = torch.cat([raw_action_p, torque.unsqueeze(-1)], dim=-1)
                 self.prosthetic_agent.action_history.update(combined_entry)

            # --- F. Update Metrics ---
            self.update_metrics_from_env_extras(
                metrics, extras, active_env_ids, active_motion_ids, prefix=True,
            )

class CoLearningOrchestrator:
    def __init__(
        self,
        fabric,
        env,
        config: Any,
        root_dir=None,
        **kwargs
        # REMOVED: agent_mappings (The agents know their own indices!)
    ):
        self.fabric = fabric
        self.env = env
        self.config = config
        self.device = fabric.device
        
        self.just_loaded_checkpoint_should_evaluate = False
        self.best_evaluated_score = None

        # --- 1. Instantiate Sub-Agents Here ---
        # We read the configs you created in 'agent_config' and spawn PPOAgents
        self.agents = {}
        
        # config.agents is the dictionary: {'humanoid': cfg, 'prosthetic': cfg}
        for agent_name, agent_cfg in config.agents.items():
            print(f"[Orchestrator] Initializing sub-agent: {agent_name}")
            # We assume both are PPOAgents based on your config
            self.agents[agent_name] = PPO(
                config=agent_cfg, 
                env=env, 
                fabric=fabric,
                root_dir=root_dir  # ← pass it through
            )

        self.num_envs: int = self.env.num_envs
        self.num_steps: int = self.config.num_steps
        self.num_mini_epochs: int = self.config.num_mini_epochs
        self.gamma: float = self.config.gamma
        self._should_stop: bool = False
        self.max_epochs: int = (
            self.config.training_max_steps
            // self.fabric.world_size
            // self.num_envs
            // self.num_steps
        )

        # Shared Training Parameters
        self.current_epoch = 0
        self.should_stop = False

        self.time_report = TimeReport()
        self.time_report.add_timer("Main Timer")

        self.evaluator = CoLearningMimicEvaluator(
            humanoid_agent=self.agents['humanoid'],
            prosthetic_agent=self.agents['prosthetic'],
            fabric=self.fabric,
            config=self.agents['humanoid'].config.evaluator # Use humanoid config
        )

    def load(self, checkpoint: Path, load_env: bool = True):
        """Load checkpoints for all sub-agents.

        Expects checkpoints to be named per-agent, e.g.:
            /path/to/run/humanoid.ckpt
            /path/to/run/prosthetic.ckpt

        Falls back to a single shared checkpoint path if agent-specific
        files are not found (useful for resuming from a single snapshot).

        Args:
            checkpoint: Path to a checkpoint file OR a directory containing
                        per-agent checkpoint files.
            load_env:   Whether to also restore environment state after loading.
        """
        if checkpoint is None:
            return

        checkpoint = Path(checkpoint).resolve()

        for agent_name, agent in self.agents.items():
            # --- Resolve per-agent checkpoint path ---
            if checkpoint.is_dir():
                # Directory mode: look for <dir>/<agent_name>.ckpt
                agent_ckpt_path = checkpoint / f"{agent_name}.ckpt"
            else:
                # File mode: look for <stem>_<agent_name><suffix>, e.g. checkpoint_humanoid.ckpt
                agent_ckpt_path = checkpoint.with_name(
                    f"{agent_name}_{checkpoint.name}"  # humanoid_ + last.ckpt
                )

            # Fall back to the exact path provided if neither variant exists
            if not agent_ckpt_path.exists():
                if checkpoint.exists():
                    print(
                        f"[Orchestrator] No agent-specific checkpoint found for "
                        f"'{agent_name}', falling back to: {checkpoint}"
                    )
                    agent_ckpt_path = checkpoint
                else:
                    print(
                        f"[Orchestrator] WARNING: No checkpoint found for "
                        f"'{agent_name}', skipping."
                    )
                    continue

            print(f"[Orchestrator] Loading '{agent_name}' from: {agent_ckpt_path}")
            state_dict = torch.load(
                agent_ckpt_path, map_location=self.device, weights_only=False
            )
            agent.load_parameters(state_dict)

        # --- Restore shared orchestrator-level state ---
        # Pull epoch/step from the first agent that has them, so the orchestrator
        # resumes at the right point rather than restarting from 0.
        first_agent = next(iter(self.agents.values()))
        self.current_epoch = first_agent.current_epoch

        # --- Restore environment state (once, shared across all agents) ---
        if load_env:
            task_id = self.env.get_task_id()
            # Root dir is taken from the checkpoint's parent directory
            env_ckpt_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
            env_checkpoint = env_ckpt_dir / f"env_{task_id}.ckpt"

            if env_checkpoint.exists():
                print(f"[Orchestrator] Loading env checkpoint: {env_checkpoint}")
                env_state_dict = torch.load(
                    env_checkpoint, map_location=self.device, weights_only=False
                )
                self.env.load_state_dict(env_state_dict)
            else:
                print(
                    f"[Orchestrator] No env checkpoint found at {env_checkpoint}, "
                    f"skipping env restore."
                )

        self.just_loaded_checkpoint_should_evaluate = True

    def _setup_agent_buffers(self):
        """Initialize buffers using the agent's own logic."""
        # 1. Get initial observations (Full Global Obs)     

        
        if self.agents['prosthetic'].save_actions:
            # Create a zero tensor matching the action shape from the buffer
            zero_action = torch.zeros(
                self.agents['prosthetic'].num_envs, 
                self.agents['prosthetic'].action_history.data.shape[-1], 
                device=self.agents['prosthetic'].device
            )
            self.agents['prosthetic'].action_history.set_all(zero_action)

            if hasattr(self.agents['prosthetic'], 'prev_command_buffer'):
                self.agents['prosthetic'].prev_command_buffer.zero_()

        global_obs, _ = self.env.reset()

        current_obs = global_obs.copy()

        # --- 3. Chain the Augmentations ---
        # A. Humanoid adds "blind_body_obs" to current_obs
        current_obs = self.agents['humanoid'].add_agent_info_to_obs(current_obs)
        
        # B. Prosthetic adds "history" and "torque" to THE SAME current_obs
        current_obs = self.agents['prosthetic'].add_agent_history_to_obs(current_obs)
        
        # C. Convert to TensorDict
        # (Assuming both agents share the same tensordict logic, calling it from one is fine)
        obs_td = self.agents['humanoid'].obs_dict_to_tensordict(current_obs)
        print("obs_td keys:", list(obs_td.keys()))


        for name, agent in self.agents.items():
            # B. Initialize Buffer
            agent.experience_buffer = ExperienceBuffer(
                agent.num_envs, agent.num_steps, device=agent.device
            )

            # C. Register Keys (Environment)
            for key, env_tensor in obs_td.items():
                agent.experience_buffer.register_key(
                    key, shape=env_tensor.shape[1:], dtype=env_tensor.dtype
                )

            # D. Register Keys (Model Output)
            with torch.no_grad():
                output_td = agent.model(obs_td.clone())
                agent.model_output_keys = agent.model.out_keys
                print(f"[{name}] model out_keys:", agent.model_output_keys)
                for key in agent.model_output_keys:
                    value = output_td[key]
                    if isinstance(value, torch.Tensor):
                        shape = value.shape[1:] if value.ndim > 1 else ()
                        agent.experience_buffer.register_key(key, shape=shape, dtype=value.dtype)
            
            # E. Register Standard Keys
            agent.experience_buffer.register_key("rewards")
            if agent.config.normalize_rewards:
                agent.experience_buffer.register_key("unnormalized_rewards")
            agent.experience_buffer.register_key("total_rewards")
            agent.experience_buffer.register_key("dones", dtype=torch.long)
            agent.register_algorithm_experience_buffer_keys()
            
            # Start Timer
            if agent.fit_start_time is None:
                agent.fit_start_time = time.time()
            self.fabric.call("on_fit_start", agent)

    def fit(self):
        self._setup_agent_buffers()

        if self.agents['prosthetic'].save_actions:
            # Create a zero tensor matching the action shape from the buffer
            zero_action = torch.zeros(
                self.agents['prosthetic'].num_envs, 
                self.agents['prosthetic'].action_history.data.shape[-1], 
                device=self.agents['prosthetic'].device
            )
        
        # Force reset on fit start
        done_indices = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
        self.time_report.start_timer('Main Timer')

        while self.current_epoch < self.max_epochs:
            self.epoch_start_time = time.time()

            # Set agents to Eval mode (freeze BatchNorm etc.)
            for agent in self.agents.values():
                agent.eval()
                agent.epoch_start_time = self.epoch_start_time
                self.fabric.call("before_play_steps", agent)

            # ===============================================================
            # 1. Data Collection Loop
            # ===============================================================
            with torch.no_grad():
                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch} (Co-Learning)",
                ):
                    # --- A. Global Reset ---
                    # Reset gives us the FULL observation dict
                    global_obs, _ = self.env.reset(done_indices)

                    # --- 3. Chain the Augmentations ---
                    # A. Humanoid adds "blind_body_obs" to current_obs
                    obs = self.agents['humanoid'].add_agent_info_to_obs(global_obs)

                    # B. Prosthetic adds "history" and "torque" to THE SAME current_obs
                    obs = self.agents['prosthetic'].add_agent_history_to_obs(obs)
        
                    # C. Convert to TensorDict
                    # (Assuming both agents share the same tensordict logic, calling it from one is fine)
                    obs_td = self.agents['humanoid'].obs_dict_to_tensordict(obs)
                    

                    # --- B. Query Agents (Parallel) ---
                    for name, agent in self.agents.items():
                        # 2. Store to Buffer
                        for key, env_tensor in obs_td.items():
                            agent.experience_buffer.update_data(key, step, env_tensor)

                        # 3. Get Action (Small dimension, e.g., 10)
                    action_h = self.agents['humanoid'].collect_rollout_step(obs_td, step)
                    self.agents['humanoid'].check_obs_for_nans(obs_td, action_h)

                    env_action = self.agents['humanoid'].expand_action_to_env(action_h)

                    action_p = self.agents['prosthetic'].collect_rollout_step_withcalc(obs_td, step)
                    env_action += self.agents['prosthetic'].expand_action_to_env(action_p)

                    # --- D. Step Environment ---
                    next_global_obs, rewards, raw_dones, raw_terminated, extras = self.env.step(env_action)

                    # -----------------------------------------------------------
                    # FUTURE-PROOFING: SYNCHRONIZED TERMINATION
                    # -----------------------------------------------------------
                    # 1. Start with Environment's decision (Physics/Time)
                    combined_dones = raw_dones.clone()
                    combined_terminated = raw_terminated.clone()

                    # 2. Collect Agent-Specific Terminations (e.g., AMP)
                    # We store these temporarily so we don't have to compute them twice
                    agent_modifications = {}

                    for name, agent in self.agents.items():
                        # Ask agent: "Do YOU want to terminate?"
                        # We pass clones so they don't modify the source tensors in-place yet
                        a_dones, a_terminated, a_extras = agent.post_env_step_modifications(
                            raw_dones.clone(), raw_terminated.clone(), extras.copy()
                        )
                        
                        # Store for later recording
                        agent_modifications[name] = {
                            "dones": a_dones,
                            "terminated": a_terminated,
                            "extras": a_extras
                        }

                        # 3. LOGICAL OR: If ANYONE says "Done", we are ALL Done.
                        combined_dones = combined_dones | a_dones
                        combined_terminated = combined_terminated | a_terminated

                    # 4. Update the Master Reset List
                    # This ensures the Environment ACTUALLY resets these indices at the start of the next loop
                    done_indices = combined_dones.nonzero(as_tuple=False).squeeze(-1)

                    
                    # -----------------------------------------------------------
                    # HANDLE REWARDS & OBSERVATIONS
                    # -----------------------------------------------------------
                    # Broadcast raw rewards (unless distinct)
                    if isinstance(rewards, dict):
                        reward_map = rewards
                    else:
                        reward_map = {name: rewards for name in self.agents.keys()}

                    # Handle History Reset (using the SYNCHRONIZED done indices)
                    if self.agents['prosthetic'].save_actions and len(done_indices) > 0:
                        self.agents['prosthetic'].action_history.set_all(
                            zero_action[done_indices], 
                            env_ids=done_indices
                        )
                        # Future-proof: Reset prev_command if it exists
                        if hasattr(self.agents['prosthetic'], 'prev_command_buffer'):
                            self.agents['prosthetic'].prev_command_buffer[done_indices] = 0

                    # Construct Next Observations
                    next_obs = self.agents['humanoid'].add_agent_info_to_obs(next_global_obs)
                    next_obs = self.agents['prosthetic'].add_agent_history_to_obs(next_obs)
                    next_obs_td = self.agents['humanoid'].obs_dict_to_tensordict(next_obs)

                    # -----------------------------------------------------------
                    # RECORD EXPERIENCE (SYNCHRONIZED)
                    # -----------------------------------------------------------
                    for name, agent in self.agents.items():
                        # Use the extras we calculated earlier (so we get the specific AMP stats)
                        current_extras = agent_modifications[name]["extras"]
                        
                        # Select correct action
                        curr_action = action_h if name == "humanoid" else action_p

                        # CRITICAL: Record using 'combined_dones' and 'combined_terminated'
                        # This ensures the Prosthetic buffer knows the Humanoid forced a reset.
                        agent.record_rollout_step(
                            next_obs_td,
                            curr_action,
                            reward_map.get(name, reward_map.get('humanoid')), 
                            combined_dones,      # <--- The Shared Decision
                            combined_terminated, # <--- The Shared Decision
                            done_indices,
                            current_extras,
                            step,
                        )
                    
                        agent.step_count += agent.get_step_count_increment()

                # End of Rollout: Calculate Returns
                for agent in self.agents.values():
                    total_rewards = agent.get_combined_experience_buffer_rewards()
                    agent.experience_buffer.batch_update_data("total_rewards", total_rewards)

            # ===============================================================
            # 2. Optimization Phase (Sequential or Parallel)
            # ===============================================================
            aggregated_log_dict = {}
            
            for name, agent in self.agents.items():
                if agent._skip_next_policy_update:
                    agent._skip_next_policy_update = False
                    agent.pre_process_dataset()
                    _ = agent.experience_buffer.make_dict()
                    log_dict = {"skipped_policy_update": 1.0}
                else:
                    log_dict = agent.optimize_model()

                for k, v in log_dict.items():
                    aggregated_log_dict[f"{name}/{k}"] = v
                
                agent.current_epoch += 1
                self.fabric.call("after_train", agent)

            self.current_epoch += 1
            aggregated_log_dict["epoch"] = self.current_epoch

            if self.evaluator is not None:
                if (
                    self.current_epoch > 0 
                    and self.current_epoch % self.evaluator.config.eval_metrics_every == 0
                ):
                    self.fabric.call("on_eval_start", self)
                    
                    # This now calls your Custom Multi-Agent Evaluator
                    eval_log_dict, evaluated_score = self.evaluator.evaluate() 
                    
                    self.fabric.call("on_eval_end", self)

                    # Save the "Best" model if this score is a new record
                    if evaluated_score is not None:
                        if (
                            self.best_evaluated_score is None
                            or evaluated_score >= self.best_evaluated_score
                        ):
                            self.best_evaluated_score = evaluated_score
                            self._save_all("best.ckpt") # Save as 'humanoid_best.ckpt'
                    
                    # Log the test results
                    aggregated_log_dict.update(eval_log_dict)

            # ===============================================================
            # RESTORED FEATURE 2: CURRICULUM
            # ===============================================================
            # Check if we have a manager that handles episode length growing
            if getattr(self.config, "max_episode_length_manager", None) is not None:
                # Ask the manager: "How long should episodes be at this epoch?"
                new_max_len = self.config.max_episode_length_manager.current_max_episode_length(
                    self.current_epoch
                )
                
                # Update the environment
                if hasattr(self.env, "max_episode_length"):
                     self.env.max_episode_length = new_max_len
            
            # Standard End-of-Epoch stuff
            self.env.on_epoch_end(self.current_epoch)
            self._handle_checkpointing()
            self._handle_logging(aggregated_log_dict)

            if self.should_stop:
                break


        self.time_report.end_timer('Main Timer')
        self.time_report.report()
        self._save_all("last.ckpt")
        for agent in self.agents.values():
            self.fabric.call("on_fit_end", agent)


    def _handle_checkpointing(self):
        """Triggers individual agent save methods."""
        should_save_epoch = (
            self.config.save_epoch_checkpoint_every is not None 
            and self.current_epoch % self.config.save_epoch_checkpoint_every == 0
        )
        should_save_last = (
            self.current_epoch % self.config.save_last_checkpoint_every == 0
        )

        if should_save_epoch:
            self._save_all(f"epoch_{self.current_epoch}.ckpt")
        
        if should_save_last:
            self._save_all("last.ckpt")

    def _save_all(self, filename):
        for name, agent in self.agents.items():
            # Save as "humanoid_last.ckpt", "leg_last.ckpt" etc.
            agent.save(checkpoint_name=f"{name}_{filename}")

    def _handle_logging(self, training_log_dict):
        # Gather stats from all agents
        log_dict = {}
        end_time = time.time()
        
        for name, agent in self.agents.items():
            # Get meters from agent
            episode_reward_dict = agent.episode_reward_meter.mean_and_clear()
            episode_length_dict = agent.episode_length_meter.mean_and_clear()
            
            prefix_dict = {
                f"{name}/info/episode_length": episode_length_dict.get("episode_length", 0),
                f"{name}/info/episode_reward": episode_reward_dict.get("episode_reward", 0),
                f"{name}/rewards/task_rewards": agent.experience_buffer.rewards.mean().item(),
            }
            log_dict.update(prefix_dict)

            # Env tensors (Extras) - Only log once or for primary agent to avoid clutter
            env_log_dict = agent.episode_env_tensors.mean_and_clear()
            if name == list(self.agents.keys())[0]: # Log env stats only once
                 log_dict.update({f"env/{k}": v for k, v in env_log_dict.items()})

        # Add training performance logs
        log_dict.update(training_log_dict)
        
        # Aggregate across distributed ranks
        aggregated_log_dict = aggregate_scalar_metrics(log_dict, self.fabric)
        self.fabric.log_dict(aggregated_log_dict)
    
    def setup(self):
        """
        Orchestrator setup:
        Instead of creating 'one' model, we iterate through all sub-agents
        and trigger their individual setup routines.
        """
        # 1. Global callbacks (Optional, if you want to signal start of init)
        self.fabric.call("on_model_init_start") 

        self._inject_obs_pipelines()

        for name, agent in self.agents.items():
            log.info(f"Setting up Agent: {name}")
            
            # Ensure dependencies are passed if they weren't in __init__
            # (Just a safety check, usually they are passed in init)
            if not getattr(agent, "env", None):
                agent.env = self.env
            if not getattr(agent, "fabric", None):
                agent.fabric = self.fabric
                
            # 2. Call the Sub-Agent's Native Setup
            # This executes the code you pasted:
            # - self.create_model()
            # - self.model.to(self.device)
            # - pass_fabric_to_running_mean_std
            # - Dummy forward pass for lazy modules
            # - self.create_optimizers()
            agent.setup()
            
        self.fabric.call("on_model_init_end")
        self.fabric.call("on_optimizer_init_end")

    def _inject_obs_pipelines(self):
        """
        Wires up the correct obs augmentation chain per agent, mirroring
        exactly what fit() does at every rollout step:

            obs = humanoid.add_agent_info_to_obs(global_obs)      # step A
            obs = prosthetic.add_agent_history_to_obs(obs)        # step B
            obs_td = humanoid.obs_dict_to_tensordict(obs)         # step C

        Each agent gets a lambda that produces the obs dict it would
        actually see during training, so lazy modules (LazyLinear,
        RunningMeanStd) materialize with the right shapes.
        """
        humanoid   = self.agents['humanoid']
        prosthetic = self.agents['prosthetic']

        humanoid._obs_pipeline = lambda raw_obs: (
            humanoid.add_agent_info_to_obs(raw_obs)
        )

        prosthetic._obs_pipeline = lambda raw_obs: (
            prosthetic.add_agent_history_to_obs(
                humanoid.add_agent_info_to_obs(raw_obs)
            )
        )

    # def load(self, path: str):
    #     """
    #     Loads checkpoints for all agents. 
    #     Assumes 'path' points to a specific file (e.g. '.../last.ckpt')
    #     and that agents saved files with prefixes (e.g. '.../humanoid_last.ckpt').
    #     """
    #     if not path:
    #         return

    #     import os
    #     path_obj = Path(path)
    #     parent_dir = path_obj.parent
    #     filename = path_obj.name  # e.g., "last.ckpt" or "epoch_100.ckpt"

    #     for name, agent in self.agents.items():
    #         # Construct the expected filename: "humanoid_" + "last.ckpt"
    #         agent_specific_filename = f"{name}_{filename}"
    #         agent_path = parent_dir / agent_specific_filename
            
    #         if agent_path.exists():
    #             log.info(f"Loading {name} from {agent_path}")
    #             agent.load(str(agent_path))
    #         else:
    #             log.warning(f"Checkpoint for {name} not found at {agent_path}. Starting fresh.")
        
    @property
    def _skip_next_policy_update(self):
        """
        Getter: Returns True if ANY sub-agent is set to skip.
        This is useful for debugging or logging.
        """
        return any(a._skip_next_policy_update for a in self.agents.values())

    @_skip_next_policy_update.setter
    def _skip_next_policy_update(self, value):
        """
        Setter: Broadcasts the skip flag to ALL sub-agents.
        This allows the main script to write 'agent._skip_next_policy_update = True'
        and have it correctly affect the Humanoid, Prosthetic, etc.
        """
        for agent in self.agents.values():
            agent._skip_next_policy_update = value
