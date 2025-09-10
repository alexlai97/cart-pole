"""
Random Agent for Cart-Pole

This is our baseline agent that takes completely random actions.
Every other RL algorithm we implement should perform better than this!

Learning Goals:
- Establish baseline performance (~20-25 steps average)
- Understand the agent interface pattern
- See how random behavior performs in Cart-Pole
- Create foundation for more sophisticated agents
"""

import gymnasium as gym
import numpy as np

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    A simple agent that takes random actions.
    
    This serves as our baseline - any RL algorithm should beat this!
    """

    def __init__(self, action_space: gym.Space):
        """Initialize the random agent."""
        super().__init__(action_space)
        self.name = "Random"

    def select_action(self, state: np.ndarray) -> int:
        """
        Select a random action.
        
        Args:
            state: Current environment state (ignored for random agent)
            
        Returns:
            Random action from action space
        """
        return self.action_space.sample()

    def get_info(self) -> dict[str, str]:
        """Get agent information."""
        return {
            "name": self.name,
            "type": "baseline",
            "parameters": "none",
            "description": "Takes completely random actions"
        }


def evaluate_random_agent(num_episodes: int = 100, render: bool = False):
    """
    Evaluate the random agent (legacy function for backward compatibility).
    
    Args:
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        
    Returns:
        Dictionary with evaluation results
    """
    from utils.agent_evaluation import run_agent_evaluation
    
    # Create environment and agent
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env.action_space)
    env.close()
    
    # Use common evaluation pipeline
    return run_agent_evaluation(
        agent=agent,
        episodes=num_episodes,
        render=render,
        save_filename="random_agent_results.json",
        compare_baseline=False  # Random is the baseline
    )


# Legacy function for backward compatibility
def print_results(results):
    """Print results (now handled by common evaluation system)."""
    from utils.agent_evaluation import print_results as common_print_results
    common_print_results(results)


if __name__ == "__main__":
    # Run evaluation using the new system
    evaluate_random_agent(num_episodes=100, render=False)
