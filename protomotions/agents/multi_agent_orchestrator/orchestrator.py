import torch
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from rich.progress import track

from lightning.fabric import Fabric
from tensordict import TensorDict

# Import necessary utilities from your existing codebase
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.utils.data import ExperienceBuffer
from protomotions.agents.utils.metering import TimeReport
from protomotions.agents.utils.training import aggregate_scalar_metrics

log = logging.getLogger(__name__)

class CoLearningOrchestrator:
    """
    Orchestrator for simultaneous multi-agent training (Co-Learning).
    
    Replaces the standard Agent.fit() loop. It manages a single environment 
    and multiple agents, handling the splitting of observations and merging 
    of actions required for them to act as a single physical entity.
    """
    def __init__(
        self,
        fabric: Fabric,
        env: BaseEnv,
        agents: Dict[str, BaseAgent],
        config: Any,
        # Mappings define how to slice the global vectors for each agent
        # Example: {'humanoid': {'obs': slice(0, 48), 'act': slice(0, 10)}}
        agent_mappings: Dict[str, Dict[str, slice]], 
    ):
        self.fabric = fabric
        self.env = env
        self.agents = agents
        self.config = config
        self.agent_mappings = agent_mappings
        self.device = fabric.device

        # Shared Training Parameters
        self.current_epoch = 0
        self.max_epochs = config.max_epochs # Assuming config has global max_epochs
        self.num_steps = config.num_steps
        self.should_stop = False

        self.time_report = TimeReport()
        self.time_report.add_timer("Main Timer")

    def _split_obs(self, global_obs: Dict, agent_name: str) -> Dict:
        """Slices the global observation dictionary for a specific agent."""
        idx = self.agent_mappings[agent_name]['obs']
        
        # Shallow copy to avoid modifying original during iteration
        agent_obs = global_obs.copy()
        
        # Assume 'obs' key holds the main feature vector (standard in IsaacLab/Gym)
        # If your env uses a different key for the main vector, adjust here.
        if 'obs' in agent_obs:
            agent_obs['obs'] = agent_obs['obs'][:, idx]
            
        return agent_obs

    def _merge_actions(self, agent_actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Merges individual agent actions into one global action tensor."""
        # Calculate total action dimension based on the mappings
        total_dim = 0
        for mapping in self.agent_mappings.values():
            s = mapping['act']
            # Assuming slice is (start, stop, step) or similar
            if s.stop is not None:
                total_dim = max(total_dim, s.stop)
        
        global_action = torch.zeros(
            (self.env.num_envs, total_dim), 
            device=self.device, 
            dtype=torch.float32
        )
        
        for name, action in agent_actions.items():
            idx = self.agent_mappings[name]['act']
            global_action[:, idx] = action
            
        return global_action

    def _setup_agent_buffers(self):
        """
        Replicates the initialization logic from BaseAgent.fit().
        Initializes ExperienceBuffer and registers keys for all agents.
        """
        # 1. Get initial observations
        global_obs, _ = self.env.reset()
        
        for name, agent in self.agents.items():
            # Prepare agent-specific observation
            raw_obs = self._split_obs(global_obs, name)
            obs = agent.add_agent_info_to_obs(raw_obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

            # 2. Initialize Experience Buffer (Logic from BaseAgent.fit)
            agent.experience_buffer = ExperienceBuffer(
                agent.num_envs, agent.num_steps, device=agent.device
            )

            # Register environment keys
            for key, env_tensor in obs_td.items():
                shape = env_tensor.shape
                dtype = env_tensor.dtype
                agent.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

            # 3. Auto-register model output keys (Logic from BaseAgent.fit)
            with torch.no_grad():
                output_td = agent.model(obs_td)
                agent.model_output_keys = agent.model.out_keys
                
                for key in agent.model_output_keys:
                    value = output_td[key]
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 1:
                            agent.experience_buffer.register_key(key)
                        else:
                            agent.experience_buffer.register_key(
                                key, shape=value.shape[1:], dtype=value.dtype
                            )
            
            log.info(f"Agent {name}: Registered keys {agent.model_output_keys}")

            # 4. Register standard keys
            agent.experience_buffer.register_key("rewards")
            if agent.config.normalize_rewards:
                agent.experience_buffer.register_key("unnormalized_rewards")
            agent.experience_buffer.register_key("total_rewards")
            agent.experience_buffer.register_key("dones", dtype=torch.long)
            agent.register_algorithm_experience_buffer_keys()
            
            # Initialize timing and start callback
            if agent.fit_start_time is None:
                agent.fit_start_time = time.time()
            self.fabric.call("on_fit_start", agent)

    def fit(self):
        """
        The main training loop replacing agent.fit().
        """
        self._setup_agent_buffers()
        
        # Force reset on fit start
        done_indices = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
        self.time_report.start_timer('Main Timer')

        while self.current_epoch < self.max_epochs:
            self.epoch_start_time = time.time()

            # Set all agents to EVAL mode for collection (freezes batchnorm stats if any)
            for agent in self.agents.values():
                agent.eval()
                agent.epoch_start_time = self.epoch_start_time # Sync time
                self.fabric.call("before_play_steps", agent)

            # ===============================================================
            # 1. Data Collection Loop
            # ===============================================================
            with torch.no_grad():
                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch} (Co-Learning)",
                ):
                    # A. Global Reset
                    global_obs, _ = self.env.reset(done_indices)
                    
                    agent_actions = {}
                    agent_obs_tds = {}

                    # B. Query All Agents
                    for name, agent in self.agents.items():
                        # 1. Prepare Obs
                        raw_obs = self._split_obs(global_obs, name)
                        obs = agent.add_agent_info_to_obs(raw_obs)
                        obs_td = agent.obs_dict_to_tensordict(obs)
                        agent_obs_tds[name] = obs_td

                        # 2. Update Buffer (Env Data)
                        for key, env_tensor in obs_td.items():
                            agent.experience_buffer.update_data(key, step, env_tensor)

                        # 3. Collect Action
                        action = agent.collect_rollout_step(obs_td, step)
                        agent.check_obs_for_nans(obs_td, action)
                        agent_actions[name] = action

                    # C. Step Environment (Merge Actions)
                    global_action = self._merge_actions(agent_actions)
                    
                    next_global_obs, rewards, dones, terminated, extras = self.env.step(global_action)
                    
                    # D. Process Results for All Agents
                    # Note: We assume shared rewards and shared termination for the body.
                    done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

                    for name, agent in self.agents.items():
                        # Prepare Next Obs
                        next_raw_obs = self._split_obs(next_global_obs, name)
                        next_obs = agent.add_agent_info_to_obs(next_raw_obs)
                        next_obs_td = agent.obs_dict_to_tensordict(next_obs)

                        # Hook for agent-specific modifications (e.g. AMP discriminator)
                        # We pass copies of dones/extras so one agent doesn't corrupt another's view
                        a_dones, a_terminated, a_extras = agent.post_env_step_modifications(
                            dones.clone(), terminated.clone(), extras.copy()
                        )
                        
                        # Record Step (Metrics + Buffer)
                        agent.record_rollout_step(
                            next_obs_td,
                            agent_actions[name],
                            rewards, # Shared Reward
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
            # 2. Optimization Phase
            # ===============================================================
            aggregated_log_dict = {}
            
            for name, agent in self.agents.items():
                if agent._skip_next_policy_update:
                    agent._skip_next_policy_update = False
                    agent.pre_process_dataset()
                    _ = agent.experience_buffer.make_dict() # Clear buffer
                    log_dict = {"skipped_policy_update": 1.0}
                else:
                    log_dict = agent.optimize_model()

                # Prefix logs with agent name
                for k, v in log_dict.items():
                    aggregated_log_dict[f"{name}/{k}"] = v
                
                agent.current_epoch += 1
                self.fabric.call("after_train", agent)

            self.current_epoch += 1
            aggregated_log_dict["epoch"] = self.current_epoch

            # ===============================================================
            # 3. Checkpointing & Logging
            # ===============================================================
            # Delegate saving to agents, but trigger based on global epoch
            self._handle_checkpointing()
            
            # Post Epoch Logging (Aggregates metrics from all agents)
            self._handle_logging(aggregated_log_dict)
            
            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self._save_all("last.ckpt")
                break

        self.time_report.end_timer('Main Timer')
        self.time_report.report()
        self._save_all("last.ckpt")
        # cleanup
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