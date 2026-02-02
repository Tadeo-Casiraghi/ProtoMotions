from dataclasses import dataclass, field
from typing import Dict, List, Optional
# Import the original config
from protomotions.agents.base_agent.config import BaseAgentConfig

@dataclass
class CoLearningConfig(BaseAgentConfig):
    """Configuration for Multi-Agent Co-Learning.
    
    Inherits global settings (max_steps, checkpointing) from BaseAgentConfig,
    but adds fields to define multiple sub-agents and their vector slices.
    """
    
    # 1. The key differentiator: A dictionary of sub-agent configurations
    #    Example: {'humanoid': PPOConfig(...), 'leg': SACConfig(...)}
    agents: Dict[str, BaseAgentConfig] = field(default_factory=dict)

    # 2. Defines how to slice the global vectors for each agent.
    #    Using List[int] because standard Python 'slice' objects generally don't work 
    #    well with config systems (Hydra/Pickle).
    #    Format: {'agent_name': {'obs': [start, end], 'act': [start, end]}}
    mapping_info: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)

    # 3. Override _target_ if you want the train script to auto-detect the Orchestrator
    #    (Optional, depending on how you modified train_agent.py)
    _target_: str = "protomotions.runners.co_learning.CoLearningOrchestrator"

    # NOTE: 'model' field from BaseAgentConfig is ignored at this top level,
    # as each agent in 'agents' dict will have its own model config.