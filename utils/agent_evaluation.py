"""
Common Agent Evaluation Utilities

This module provides generic evaluation functions that work with any agent
implementing the BaseAgent interface, eliminating code duplication.
"""

import json
import os
from typing import Any

import gymnasium as gym
import numpy as np

from agents.base_agent import BaseAgent


def evaluate_agent(agent: BaseAgent, num_episodes: int = 100, render: bool = False) -> dict[str, Any]:
    """
    Evaluate any agent over multiple episodes.
    
    This is a generic evaluation function that works with any agent
    implementing the BaseAgent interface.
    
    Args:
        agent: The agent to evaluate (must inherit from BaseAgent)
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"🤖 Evaluating {agent.name} Agent over {num_episodes} episodes...")

    # Create environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    # Track performance
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Agent selects action
            action = agent.select_action(state)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"  Episodes {episode-19:3d}-{episode+1:3d}: Average = {avg_reward:5.1f} steps")

    env.close()

    # Calculate statistics
    results = {
        "agent": agent.get_info(),
        "episodes": num_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "success_rate": np.mean(np.array(episode_rewards) >= 195),  # CartPole solved = 195+ avg
    }

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print evaluation results in a nice format."""
    agent_name = results["agent"]["name"]
    agent_emoji = get_agent_emoji(agent_name)
    
    print("\n" + "="*50)
    print(f"{agent_emoji} {agent_name} Agent Results")
    print("="*50)
    print(f"Episodes run: {results['episodes']}")
    print(f"Average performance: {results['mean_reward']:.2f} ± {results['std_reward']:.2f} steps")
    print(f"Range: {results['min_reward']:.0f} - {results['max_reward']:.0f} steps")
    print(f"Success rate: {results['success_rate']:.1%} (episodes ≥ 195 steps)")
    print()

    # Performance interpretation
    performance_level = interpret_performance(results['mean_reward'])
    print(f"{performance_level['emoji']} Performance: {performance_level['description']}")

    print(f"\n💡 Learning Insight:")
    print(f"This {agent_name.lower()} approach shows {performance_level['insight']}")
    print(f"Performance of {results['mean_reward']:.1f} steps {performance_level['context']}")


def get_agent_emoji(agent_name: str) -> str:
    """Get emoji for agent type."""
    emoji_map = {
        "Random": "🎲",
        "Rule-Based": "🧠",
        "Q-Learning": "📚",
        "DQN": "🧠",
        "REINFORCE": "🎯",
        "A2C": "🎭",
        "PPO": "🚀"
    }
    return emoji_map.get(agent_name, "🤖")


def interpret_performance(mean_reward: float) -> dict[str, str]:
    """Interpret performance level based on mean reward."""
    if mean_reward < 50:
        return {
            "emoji": "🔴",
            "description": "Poor (needs significant improvement)",
            "insight": "the challenge of this environment",
            "context": "reveals room for major improvements."
        }
    elif mean_reward < 100:
        return {
            "emoji": "🟡", 
            "description": "Below average (making progress)",
            "insight": "incremental learning progress",
            "context": "shows the agent is learning but needs more training."
        }
    elif mean_reward < 195:
        return {
            "emoji": "🟠",
            "description": "Good but not solved (close!)",
            "insight": "strong performance approaching mastery",
            "context": "demonstrates solid understanding of the task."
        }
    else:
        return {
            "emoji": "🟢",
            "description": "Solved! (excellent performance)",
            "insight": "mastery of the Cart-Pole environment",
            "context": "has successfully learned to balance the pole!"
        }


def save_results(results: dict[str, Any], filename: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary from evaluate_agent()
        filename: Filename (will be saved in outputs/results/)
    """
    os.makedirs("outputs/results", exist_ok=True)
    filepath = f"outputs/results/{filename}"
    
    # Convert numpy types to native Python for JSON serialization
    json_results = results.copy()
    for key in ["episode_rewards", "episode_lengths"]:
        if key in json_results:
            json_results[key] = [float(x) for x in json_results[key]]
    
    for key in ["mean_reward", "std_reward", "min_reward", "max_reward", "success_rate"]:
        if key in json_results:
            json_results[key] = float(json_results[key])

    with open(filepath, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to {filepath}")


def compare_agents(agent_results: dict[str, Any], baseline_file: str = "random_agent_results.json") -> None:
    """
    Compare agent performance with baseline.
    
    Args:
        agent_results: Current agent results
        baseline_file: Baseline results filename to compare against
    """
    baseline_path = f"outputs/results/{baseline_file}"
    
    if not os.path.exists(baseline_path):
        print(f"\n⚠️  Baseline results not found at {baseline_path}")
        print("Run the baseline agent first for comparison.")
        return
    
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    
    current_name = agent_results["agent"]["name"]
    baseline_name = baseline_results["agent"]["name"]
    
    print("\n" + "="*50)
    print(f"📊 COMPARISON: {current_name} vs {baseline_name}")
    print("="*50)
    print(f"{baseline_name} Agent:     {baseline_results['mean_reward']:.2f} ± {baseline_results['std_reward']:.2f} steps")
    print(f"{current_name} Agent: {agent_results['mean_reward']:.2f} ± {agent_results['std_reward']:.2f} steps")
    
    improvement = agent_results['mean_reward'] - baseline_results['mean_reward']
    improvement_pct = (improvement / baseline_results['mean_reward']) * 100
    
    print(f"\n🎯 Improvement: {improvement:+.2f} steps ({improvement_pct:+.1f}%)")
    
    if improvement > 10:
        print(f"✅ SUCCESS: {current_name} significantly beats {baseline_name.lower()}!")
    elif improvement > 0:
        print(f"✅ MODEST WIN: {current_name} is better than {baseline_name.lower()}")
    else:
        print(f"❌ NEEDS WORK: {current_name} no better than {baseline_name.lower()}")


def run_agent_evaluation(agent: BaseAgent, episodes: int = 100, render: bool = False, 
                        save_filename: str = None, compare_baseline: bool = True) -> dict[str, Any]:
    """
    Complete evaluation pipeline for any agent.
    
    Args:
        agent: Agent to evaluate
        episodes: Number of episodes
        render: Whether to render
        save_filename: Filename to save results (auto-generated if None)
        compare_baseline: Whether to compare with random baseline
        
    Returns:
        Evaluation results
    """
    # Evaluate agent
    results = evaluate_agent(agent, episodes, render)
    
    # Print results
    print_results(results)
    
    # Save results
    if save_filename is None:
        save_filename = f"{agent.name.lower().replace('-', '_')}_agent_results.json"
    save_results(results, save_filename)
    
    # Compare with baseline if requested
    if compare_baseline:
        compare_agents(results)
    
    return results