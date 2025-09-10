"""
Visualization tools for analyzing agent performance.

This module creates plots and charts to understand how different agents perform,
including learning curves, performance distributions, and comparisons.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_agent_results(results_file: str) -> dict[str, Any]:
    """Load agent results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def plot_episode_rewards(results: dict[str, Any], save_path: str = None) -> None:
    """
    Plot episode rewards over time for an agent.
    
    Args:
        results: Agent evaluation results
        save_path: Optional path to save the plot
    """
    episode_rewards = results["episode_rewards"]
    agent_name = results["agent"]["name"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Episode rewards over time
    ax1.plot(episode_rewards, alpha=0.6, linewidth=0.8, color='steelblue')

    # Add rolling average
    window = min(20, len(episode_rewards) // 5)
    if window > 1:
        rolling_avg = np.convolve(
            episode_rewards,
            np.ones(window)/window,
            mode='valid'
        )
        ax1.plot(
            range(window-1, len(episode_rewards)),
            rolling_avg,
            color='red',
            linewidth=2,
            label=f'Rolling avg ({window} episodes)'
        )

    ax1.axhline(y=results["mean_reward"], color='green', linestyle='--',
               label=f'Overall avg: {results["mean_reward"]:.1f}')
    ax1.axhline(y=195, color='orange', linestyle='--', alpha=0.7,
               label='Solved threshold (195)')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward (Steps Survived)')
    ax1.set_title(f'{agent_name} Agent: Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reward distribution histogram
    ax2.hist(episode_rewards, bins=20, alpha=0.7, color='steelblue',
            edgecolor='black')
    ax2.axvline(x=results["mean_reward"], color='green', linestyle='--',
               linewidth=2, label=f'Mean: {results["mean_reward"]:.1f}')
    ax2.axvline(x=195, color='orange', linestyle='--', alpha=0.7,
               linewidth=2, label='Solved (195)')

    ax2.set_xlabel('Episode Reward (Steps)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{agent_name} Agent: Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")

    plt.show()


def plot_performance_summary(results: dict[str, Any], save_path: str = None) -> None:
    """
    Create a comprehensive performance summary plot.
    
    Args:
        results: Agent evaluation results  
        save_path: Optional path to save the plot
    """
    agent_name = results["agent"]["name"]
    episode_rewards = np.array(results["episode_rewards"])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Basic statistics
    stats = [
        results["mean_reward"],
        results["std_reward"],
        results["min_reward"],
        results["max_reward"]
    ]
    stat_names = ['Mean', 'Std Dev', 'Min', 'Max']
    colors = ['steelblue', 'orange', 'red', 'green']

    bars = ax1.bar(stat_names, stats, color=colors, alpha=0.7)
    ax1.set_ylabel('Steps')
    ax1.set_title(f'{agent_name} Agent: Performance Statistics')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, stat in zip(bars, stats, strict=False):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{stat:.1f}', ha='center', va='bottom')

    # Plot 2: Success rate pie chart
    success_episodes = np.sum(episode_rewards >= 195)
    fail_episodes = len(episode_rewards) - success_episodes

    if success_episodes > 0:
        ax2.pie([success_episodes, fail_episodes],
               labels=['Success (≥195)', 'Failed (<195)'],
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%', startangle=90)
    else:
        ax2.pie([fail_episodes], labels=['Failed (<195)'],
               colors=['lightcoral'], autopct='%1.1f%%')

    ax2.set_title(f'{agent_name} Agent: Success Rate')

    # Plot 3: Learning curve with confidence intervals
    episodes = np.arange(len(episode_rewards))

    # Calculate rolling statistics
    window = min(10, len(episode_rewards) // 10)
    if window > 1:
        rolling_mean = np.convolve(episode_rewards, np.ones(window)/window, mode='same')
        rolling_std = np.array([
            np.std(episode_rewards[max(0, i-window//2):i+window//2+1])
            for i in range(len(episode_rewards))
        ])

        ax3.plot(episodes, rolling_mean, color='blue', linewidth=2, label='Rolling mean')
        ax3.fill_between(episodes,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.3, color='blue', label='±1 std dev')
    else:
        ax3.plot(episodes, episode_rewards, color='blue', alpha=0.6)

    ax3.axhline(y=195, color='orange', linestyle='--', label='Solved (195)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward (Steps)')
    ax3.set_title(f'{agent_name} Agent: Performance Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    perc_values = np.percentile(episode_rewards, percentiles)

    ax4.plot(percentiles, perc_values, 'o-', color='purple', linewidth=2,
            markersize=6)
    ax4.axhline(y=195, color='orange', linestyle='--', alpha=0.7,
               label='Solved (195)')
    ax4.set_xlabel('Percentile')
    ax4.set_ylabel('Reward (Steps)')
    ax4.set_title(f'{agent_name} Agent: Performance Percentiles')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle(f'{agent_name} Agent: Comprehensive Performance Analysis',
                fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Summary plot saved to {save_path}")

    plt.show()


def analyze_random_agent() -> None:
    """Analyze the random agent results and create visualizations."""
    print("📊 Analyzing Random Agent Performance...")

    results_file = "outputs/results/random_agent_results.json"

    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("Run 'python agents/random_agent.py' first to generate results.")
        return

    # Load results
    results = load_agent_results(results_file)

    # Create output directory for plots
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_episode_rewards(
        results,
        save_path=str(plots_dir / "random_agent_episodes.png")
    )

    plot_performance_summary(
        results,
        save_path=str(plots_dir / "random_agent_summary.png")
    )

    print("\n✅ Random agent analysis complete!")
    print(f"📁 Plots saved in {plots_dir}/")


if __name__ == "__main__":
    analyze_random_agent()
