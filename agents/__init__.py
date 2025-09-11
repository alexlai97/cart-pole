"""
Reinforcement learning agents for Cart-Pole environment.

This package contains implementations of various RL algorithms:
- Random agent (baseline)
- Rule-based agent (heuristic)
- Q-Learning (tabular) 
- Deep Q-Network (DQN)
- REINFORCE (policy gradient)
- Actor-Critic (A2C)
- Proximal Policy Optimization (PPO)
"""

from .base_agent import BaseAgent
from .random_agent import RandomAgent, evaluate_random_agent
from .rule_based_agent import RuleBasedAgent, evaluate_rule_based_agent
from .q_learning_agent import QLearningAgent, train_q_learning_agent

# Deep learning agents (require torch) - use lazy imports
try:
    from .dqn_agent import DQNAgent, DQNetwork, train_dqn_agent
    _HAS_TORCH = True
except ImportError as e:
    _HAS_TORCH = False
    
    # Provide helpful error message if someone tries to use DQN without torch
    class _MissingDependencyAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"DQN agent requires PyTorch. Install it with: pip install torch>=2.0.0\n"
                f"Or install all deep learning dependencies: pip install '.[deep]'\n"
                f"Original error: {e}"
            )
    
    DQNAgent = _MissingDependencyAgent
    DQNetwork = _MissingDependencyAgent  
    train_dqn_agent = _MissingDependencyAgent
