"""
Centralized constants for Cart-Pole RL project.

This module contains shared constants used across agents for comparisons,
baselines, and project configuration.
"""

from typing import Dict, Tuple

# Agent performance baselines (mean, std)
BASELINES: Dict[str, Tuple[float, float]] = {
    "random": (23.3, 11.5),
    "rule_based": (43.8, 8.7), 
    "q_learning": (28.5, 12.8),
}

# Success thresholds
CART_POLE_SUCCESS_THRESHOLD = 195  # CartPole-v1 is considered solved at 195+ steps
EPISODE_CAP = 500  # Maximum steps per episode

# Display names for agents
AGENT_DISPLAY_NAMES: Dict[str, str] = {
    "random": "Random Agent",
    "rule_based": "Rule-Based Agent",
    "q_learning": "Q-Learning Agent", 
    "dqn": "DQN Agent",
}

def format_baseline_comparison(agent_name: str, mean_reward: float, std_reward: float) -> str:
    """Format a baseline comparison string for consistent display."""
    lines = ["🎯 Agent Performance vs Baselines:"]
    
    for baseline_name, (baseline_mean, baseline_std) in BASELINES.items():
        lines.append(f"   {AGENT_DISPLAY_NAMES[baseline_name]}: {baseline_mean:.1f} ± {baseline_std:.1f} steps")
    
    lines.append(f"   {AGENT_DISPLAY_NAMES.get(agent_name, agent_name.title())}: {mean_reward:.1f} ± {std_reward:.1f} steps")
    
    return "\n".join(lines)

def calculate_improvement(current_mean: float, baseline_name: str) -> float:
    """Calculate percentage improvement over a baseline agent."""
    if baseline_name not in BASELINES:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    
    baseline_mean, _ = BASELINES[baseline_name]
    return ((current_mean - baseline_mean) / baseline_mean) * 100