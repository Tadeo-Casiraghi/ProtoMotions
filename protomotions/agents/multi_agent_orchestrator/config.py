from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Assuming BaseAgentConfig defines max_steps, batch_size, etc.
from protomotions.agents.base_agent.config import BaseAgentConfig 
from protomotions.agents.ppo.config import PPOAgentConfig

@dataclass
class CoLearningConfig(BaseAgentConfig):
    """
    Orchestrator config. 
    It manages the training loop (fit function) and holds the sub-agents.
    """
    
    # 1. Point to your new Orchestrator class
    _target_: str = "protomotions.agents.multi_agent_orchestrator.orchestrator.CoLearningOrchestrator"

    # 2. The sub-agents. 
    # The keys (e.g., 'humanoid', 'prosthetic') will act as IDs.
    agents: Dict[str, PPOAgentConfig] = field(default_factory=dict)

    # 4. Global constraints (Optional)
    # If you need to sync updates, you might want this here
    sync_updates: bool = True