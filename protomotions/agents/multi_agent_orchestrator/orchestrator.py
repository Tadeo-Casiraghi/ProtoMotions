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
        global_obs, _ = self.env.reset()
        
        for name, agent in self.agents.items():
            # A. Process Obs (Agent filters what it needs internally)
            # We copy global_obs so one agent doesn't mutate it for the other
            obs = agent.add_agent_info_to_obs(global_obs.copy())
            obs_td = agent.obs_dict_to_tensordict(obs)

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
        # TODO: Check if its correct
        self._setup_agent_buffers()
        
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
                    
                    expanded_actions_list = []
                    agent_specific_data = {} # To store data needed for recording later

                    # --- B. Query Agents (Parallel) ---
                    for name, agent in self.agents.items():
                        # 1. Process Obs: Pass the FULL global_obs. 
                        # agent.add_agent_info_to_obs will filter out the blind bodies.
                        obs = agent.add_agent_info_to_obs(global_obs.copy())
                        obs_td = agent.obs_dict_to_tensordict(obs)

                        # 2. Store to Buffer
                        for key, env_tensor in obs_td.items():
                            agent.experience_buffer.update_data(key, step, env_tensor)

                        # 3. Get Action (Small dimension, e.g., 10)
                        action = agent.collect_rollout_step(obs_td, step)
                        agent.check_obs_for_nans(obs_td, action)

                        # 4. Expand Action (Full dimension, e.g., 50, mostly zeros)
                        # This uses the helper function you provided!
                        full_env_action = agent.expand_action_to_env(action)
                        expanded_actions_list.append(full_env_action)

                        # 5. Store temp data for the record phase
                        agent_specific_data[name] = {
                            "obs_td": obs_td,  # Needed for next step recording? Usually we record *next* obs or current? 
                            # Standard PPO records (next_obs, action, reward, done)
                            "action": action 
                        }

                    # --- C. Merge Actions ---
                    # Since expand_action_to_env puts 0.0 in non-active joints, we can just SUM them.
                    # e.g. Agent A: [1, 1, 0, 0] + Agent B: [0, 0, 1, 1] = [1, 1, 1, 1]
                    global_action = torch.stack(expanded_actions_list).sum(dim=0)
                    
                    # --- D. Step Environment ---
                    next_global_obs, rewards, dones, terminated, extras = self.env.step(global_action)
                    
                    # Handle Reward Distribution
                    # If rewards is a dict {'humanoid': ..., 'prosthetic': ...}, split it.
                    # If it's a single tensor, share it.
                    if isinstance(rewards, dict):
                        reward_map = rewards
                    else:
                        # Broadcast shared reward to all agents
                        reward_map = {name: rewards for name in self.agents.keys()}

                    done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

                    # --- E. Record Experience ---
                    for name, agent in self.agents.items():
                        # 1. Process NEXT Obs
                        next_obs = agent.add_agent_info_to_obs(next_global_obs.copy())
                        next_obs_td = agent.obs_dict_to_tensordict(next_obs)

                        # 2. Agent-specific modifications (AMP etc)
                        a_dones, a_terminated, a_extras = agent.post_env_step_modifications(
                            dones.clone(), terminated.clone(), extras.copy()
                        )
                        
                        # 3. Record
                        # Note: We use the *small* action here (agent_specific_data[name]["action"])
                        # because PPO needs to calculate log_prob of the *small* action later.
                        agent.record_rollout_step(
                            next_obs_td,
                            agent_specific_data[name]["action"],
                            reward_map.get(name, reward_map.get('humanoid')), # Fallback safely
                            a_dones,
                            a_terminated,
                            done_indices,
                            a_extras,
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

            # ===============================================================
            # 3. Checkpointing & Logging
            # ===============================================================
            self._handle_checkpointing()
            self._handle_logging(aggregated_log_dict)
            
            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self._save_all("last.ckpt")
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