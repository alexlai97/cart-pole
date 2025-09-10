"""
Base Agent Class for Cart-Pole RL Project

This module defines the abstract base class that all agents must implement,
ensuring consistent interface and reducing code duplication.
"""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all Cart-Pole agents.
    
    This defines the common interface that all agents must implement,
    ensuring consistency and enabling generic evaluation functions.
    """

    def __init__(self, action_space: gym.Space):
        """
        Initialize the agent.
        
        Args:
            action_space: The environment's action space
        """
        self.action_space = action_space
        self.name: str = "BaseAgent"  # Should be overridden by subclasses

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current environment state
            
        Returns:
            Action to take (integer)
        """
        pass

    def train(self, *args, **kwargs) -> None:
        """
        Train the agent (default: no training for non-learning agents).
        
        Learning agents should override this method to implement their
        specific training logic.
        """
        pass

    def save(self, filepath: str) -> None:
        """
        Save agent to file (default: nothing to save).
        
        Agents with learnable parameters should override this method.
        """
        print(f"{self.name} agent has nothing to save to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load agent from file (default: nothing to load).
        
        Agents with learnable parameters should override this method.
        """
        print(f"{self.name} agent has nothing to load from {filepath}")

    def get_info(self) -> dict[str, Any]:
        """
        Get agent information for logging and comparison.
        
        Returns:
            Dictionary with agent metadata
        """
        return {
            "name": self.name,
            "type": "base",
            "parameters": "none",
            "description": "Base agent class"
        }