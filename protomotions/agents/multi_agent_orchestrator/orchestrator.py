import torch
import time
import logging
from typing import Dict, Any
from rich.progress import track

from pathlib import Path
from protomotions.agents.utils.data import ExperienceBuffer
from protomotions.agents.utils.metering import TimeReport
from protomotions.agents.utils.training import aggregate_scalar_metrics

log = logging.getLogger(__name__)

class CoLearningOrchestrator:
    def __init__(
        self,
        fabric,
        env,
        agents: Dict[str, Any],
        config: Any,
        # REMOVED: agent_mappings (The agents know their own indices!)
    ):
        self.fabric = fabric
        self.env = env
        self.agents = agents
        self.config = config
        self.device = fabric.device

        # Shared Training Parameters
        self.current_epoch = 0
        self.max_epochs = config.max_epochs 
        self.num_steps = config.num_steps
        self.should_stop = False

        self.time_report = TimeReport()
        self.time_report.add_timer("Main Timer")

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
                output_td = agent.model(obs_td)
                agent.model_output_keys = agent.model.out_keys
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

            if getattr(self, "evaluator", None) is not None:
                if (
                    self.current_epoch > 0 
                    and self.current_epoch % self.evaluator.config.eval_metrics_every == 0
                ):
                    self.fabric.call("on_eval_start", self)
                    
                    # Run the test loop (You need to implement this for multi-agent!)
                    # It looks just like the training loop but without .backward()
                    eval_log_dict, evaluated_score = self.evaluate() 
                    
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

    def load(self, path: str):
        """
        Loads checkpoints for all agents. 
        Assumes 'path' points to a specific file (e.g. '.../last.ckpt')
        and that agents saved files with prefixes (e.g. '.../humanoid_last.ckpt').
        """
        if not path:
            return

        import os
        path_obj = Path(path)
        parent_dir = path_obj.parent
        filename = path_obj.name  # e.g., "last.ckpt" or "epoch_100.ckpt"

        for name, agent in self.agents.items():
            # Construct the expected filename: "humanoid_" + "last.ckpt"
            agent_specific_filename = f"{name}_{filename}"
            agent_path = parent_dir / agent_specific_filename
            
            if agent_path.exists():
                log.info(f"Loading {name} from {agent_path}")
                agent.load(str(agent_path))
            else:
                log.warning(f"Checkpoint for {name} not found at {agent_path}. Starting fresh.")
        
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

    def evaluate(self):
        # TODO: Check what the heck
        """
        Runs a Multi-Agent evaluation loop.
        Returns:
            log_dict (dict): Metrics to log (e.g. average reward).
            score (float): The primary score used to determine 'best.ckpt'.
        """
        # 1. Set all agents to Eval mode (Deterministic)
        for agent in self.agents.values():
            agent.eval()
        
        # 2. Setup Metrics
        total_reward = 0.0
        num_episodes = 0
        num_steps = 0
        
        # We'll run for a fixed number of steps or episodes
        # (e.g., 2000 steps roughly equals 5-10 seconds of walking)
        eval_steps = getattr(self.config, "eval_steps", 2000)
        
        # Force a fresh reset for evaluation
        # We typically use a specific subset of environments or just all of them
        done_indices = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
        
        # Zero out history for the prosthetic
        if self.agents['prosthetic'].save_actions:
             zero_action = torch.zeros(
                self.agents['prosthetic'].num_envs, 
                self.agents['prosthetic'].action_history.data.shape[-1], 
                device=self.agents['prosthetic'].device
            )
             self.agents['prosthetic'].action_history.set_all(zero_action)
             if hasattr(self.agents['prosthetic'], 'prev_command_buffer'):
                self.agents['prosthetic'].prev_command_buffer.zero_()

        with torch.no_grad():
            # Initial Reset
            global_obs, _ = self.env.reset(done_indices)
            
            # Chain Augmentations (Same as training)
            obs = self.agents['humanoid'].add_agent_info_to_obs(global_obs)
            obs = self.agents['prosthetic'].add_agent_history_to_obs(obs)
            obs_td = self.agents['humanoid'].obs_dict_to_tensordict(obs)

            for step in range(eval_steps):
                # --- Get Deterministic Actions ---
                # Humanoid (Mode = Mean Action usually)
                # Note: collect_rollout_step usually samples. 
                # For eval, we often want the MEAN (deterministic).
                # If your agent doesn't have a 'get_action_mean' method, 
                # standard sampling with eval() mode (std=0) works too.
                action_h = self.agents['humanoid'].collect_rollout_step(obs_td, step) 
                
                # Prosthetic
                action_p = self.agents['prosthetic'].collect_rollout_step_withcalc(obs_td, step)
                
                # Merge
                env_action = self.agents['humanoid'].expand_action_to_env(action_h)
                env_action += self.agents['prosthetic'].expand_action_to_env(action_p)
                
                # --- Step ---
                next_global_obs, rewards, raw_dones, raw_terminated, extras = self.env.step(env_action)
                
                # Update Metrics
                # If rewards is dict, sum them up or pick a primary one
                if isinstance(rewards, dict):
                    step_reward = sum(r.mean().item() for r in rewards.values())
                else:
                    step_reward = rewards.mean().item()
                total_reward += step_reward

                # Handle Resets (Logic from Training)
                combined_dones = raw_dones.clone()
                for name, agent in self.agents.items():
                    a_dones, _, _ = agent.post_env_step_modifications(
                        raw_dones.clone(), raw_terminated.clone(), extras.copy()
                    )
                    combined_dones = combined_dones | a_dones
                
                done_indices = combined_dones.nonzero(as_tuple=False).squeeze(-1)
                num_episodes += len(done_indices)
                
                # Reset History if needed
                if self.agents['prosthetic'].save_actions and len(done_indices) > 0:
                     self.agents['prosthetic'].action_history.set_all(zero_action[done_indices], env_ids=done_indices)
                     if hasattr(self.agents['prosthetic'], 'prev_command_buffer'):
                        self.agents['prosthetic'].prev_command_buffer[done_indices] = 0

                # Prepare for next step
                next_obs = self.agents['humanoid'].add_agent_info_to_obs(next_global_obs)
                next_obs = self.agents['prosthetic'].add_agent_history_to_obs(next_obs)
                obs_td = self.agents['humanoid'].obs_dict_to_tensordict(next_obs)
                
                num_steps += 1

        # Calculate Results
        avg_reward = total_reward / num_steps
        
        log_dict = {
            "eval/avg_reward": avg_reward,
            "eval/num_episodes": num_episodes
        }
        
        # Return log_dict and the score (avg_reward) for checkpointing
        return log_dict, avg_reward