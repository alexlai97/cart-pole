"""
Rule-Based Agent for Cart-Pole

This agent uses simple heuristics to make decisions:
- If pole tilts right (positive angle), move right to "catch" it
- If pole tilts left (negative angle), move left to "catch" it

Learning Goals:
- Can simple rules beat random performance?
- Understand the Cart-Pole state space (position, velocity, angle, angular_velocity)
- See how domain knowledge can guide action selection
- Compare rule-based vs learning-based approaches
"""

import gymnasium as gym
import numpy as np

from agents.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    A simple agent that uses pole angle to decide actions.
    
    Strategy: Move in the direction the pole is tilting to try to "catch" it.
    This mimics how humans would intuitively play Cart-Pole.
    """

    def __init__(self, action_space: gym.Space):
        """Initialize the rule-based agent."""
        super().__init__(action_space)
        self.name = "Rule-Based"
        
        # Cart-Pole actions: 0 = push left, 1 = push right
        self.LEFT = 0
        self.RIGHT = 1

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action based on pole angle.
        
        Cart-Pole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        - pole_angle: positive = tilting right, negative = tilting left
        - Strategy: Move in direction of tilt to "catch" the falling pole
        
        Args:
            state: Current environment state [x, x_dot, theta, theta_dot]
            
        Returns:
            Action: 0 (left) or 1 (right)
        """
        # Extract pole angle (state[2])
        pole_angle = state[2]
        
        # Simple rule: move in the direction the pole is tilting
        if pole_angle > 0:  # Pole tilting right
            return self.RIGHT  # Move right to catch it
        else:  # Pole tilting left (or perfectly balanced)
            return self.LEFT  # Move left to catch it

    def get_info(self) -> dict[str, str]:
        """Get agent information."""
        return {
            "name": self.name,
            "type": "rule-based",
            "parameters": "pole_angle_threshold=0",
            "description": "Moves in direction of pole tilt to catch falling pole"
        }


def evaluate_rule_based_agent(num_episodes: int = 100, render: bool = False):
    """
    Evaluate the rule-based agent (legacy function for backward compatibility).
    
    Args:
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        
    Returns:
        Dictionary with evaluation results
    """
    from utils.agent_evaluation import run_agent_evaluation
    
    # Create environment and agent
    env = gym.make("CartPole-v1")
    agent = RuleBasedAgent(env.action_space)
    env.close()
    
    # Use common evaluation pipeline
    return run_agent_evaluation(
        agent=agent,
        episodes=num_episodes,
        render=render,
        save_filename="rule_based_agent_results.json",
        compare_baseline=True  # Compare with random baseline
    )


# Legacy functions for backward compatibility
def print_results(results):
    """Print results (now handled by common evaluation system)."""
    from utils.agent_evaluation import print_results as common_print_results
    common_print_results(results)


def compare_with_random(results):
    """Compare with random (now handled by common evaluation system)."""
    from utils.agent_evaluation import compare_agents
    compare_agents(results, "random_agent_results.json")


if __name__ == "__main__":
    # Run evaluation using the new system
    evaluate_rule_based_agent(num_episodes=100, render=False)