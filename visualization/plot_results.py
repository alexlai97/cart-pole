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
    """Load agent results from JSON file, handling different schemas."""
    with open(results_file) as f:
        data = json.load(f)
    
    # Handle different result file schemas
    if "agent" not in data:
        # Legacy or DQN flat schema - synthesize agent info
        agent_name = "DQN Agent" if "agent_name" in data and data["agent_name"] == "DQN" else "Unknown Agent"
        
        # Wrap in standard schema
        data["agent"] = {
            "name": agent_name,
            "type": data.get("agent_name", "unknown").lower()
        }
    
    return data


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


def discover_available_agents() -> list[str]:
    """Discover all available agent result files."""
    results_dir = Path("outputs/results")
    if not results_dir.exists():
        return []
    
    agent_names = []
    for file_path in results_dir.glob("*_results.json"):
        # Extract agent name from filename (remove _results.json suffix)
        agent_name = file_path.stem.replace("_results", "")
        agent_names.append(agent_name)
    
    return sorted(agent_names)


def analyze_agent(agent_name: str) -> bool:
    """Analyze a specific agent and create visualizations.
    
    Args:
        agent_name: Name of the agent (e.g., 'random', 'rule_based')
        
    Returns:
        True if analysis was successful, False otherwise
    """
    print(f"📊 Analyzing {agent_name.title().replace('_', ' ')} Agent Performance...")

    # Try multiple filename patterns for backward compatibility
    possible_files = [
        f"outputs/results/{agent_name}_results.json",
        f"outputs/results/{agent_name}_agent_results.json"
    ]
    
    results_file = None
    for file_path in possible_files:
        if Path(file_path).exists():
            results_file = file_path
            break

    if results_file is None:
        print(f"❌ Results file not found for agent '{agent_name}'")
        print(f"Looked for: {', '.join(possible_files)}")
        print(f"Run the {agent_name} agent first to generate results.")
        return False

    # Load results
    results = load_agent_results(results_file)

    # Create output directory for plots
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_episode_rewards(
        results,
        save_path=str(plots_dir / f"{agent_name}_episodes.png")
    )

    plot_performance_summary(
        results,
        save_path=str(plots_dir / f"{agent_name}_summary.png")
    )

    print(f"\n✅ {agent_name.title().replace('_', ' ')} agent analysis complete!")
    print(f"📁 Plots saved in {plots_dir}/")
    return True


def analyze_all_agents() -> None:
    """Analyze all available agents."""
    available_agents = discover_available_agents()
    
    if not available_agents:
        print("❌ No agent results found in outputs/results/")
        print("Run some agents first to generate results.")
        return
        
    print(f"📊 Found {len(available_agents)} agent(s): {', '.join(available_agents)}")
    print("\n" + "="*60)
    
    success_count = 0
    for agent_name in available_agents:
        print()
        if analyze_agent(agent_name):
            success_count += 1
        print("="*60)
    
    print(f"\n🎉 Completed analysis for {success_count}/{len(available_agents)} agents!")


def compare_agents(agent_names: list[str]) -> None:
    """Compare multiple agents side-by-side.
    
    Args:
        agent_names: List of agent names to compare
    """
    print(f"📊 Comparing {len(agent_names)} agents: {', '.join(agent_names)}")
    
    # Load all agent results
    agent_results = {}
    for agent_name in agent_names:
        # Try multiple filename patterns for backward compatibility
        possible_files = [
            f"outputs/results/{agent_name}_results.json",
            f"outputs/results/{agent_name}_agent_results.json"
        ]
        
        results_file = None
        for file_path in possible_files:
            if Path(file_path).exists():
                results_file = file_path
                break
        
        if results_file:
            agent_results[agent_name] = load_agent_results(results_file)
        else:
            print(f"⚠️  Skipping {agent_name} - results file not found")
            print(f"   Looked for: {', '.join(possible_files)}")
    
    if not agent_results:
        print("❌ No valid agent results found for comparison")
        return
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['steelblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    # Plot 1: Episode rewards comparison
    for i, (agent_name, results) in enumerate(agent_results.items()):
        episode_rewards = results["episode_rewards"]
        color = colors[i % len(colors)]
        
        # Plot raw episodes with transparency
        ax1.plot(episode_rewards, alpha=0.4, color=color, linewidth=0.8)
        
        # Add rolling average
        window = min(20, len(episode_rewards) // 5)
        if window > 1:
            rolling_avg = np.convolve(
                episode_rewards, np.ones(window)/window, mode='valid'
            )
            ax1.plot(
                range(window-1, len(episode_rewards)), rolling_avg,
                color=color, linewidth=2, label=f'{agent_name} (avg: {results["mean_reward"]:.1f})'
            )
        else:
            ax1.plot(episode_rewards, color=color, linewidth=1, 
                    label=f'{agent_name} (avg: {results["mean_reward"]:.1f})')
    
    ax1.axhline(y=195, color='black', linestyle='--', alpha=0.7, label='Solved (195)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward (Steps)')
    ax1.set_title('Episode Rewards Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance statistics comparison
    agent_names_list = list(agent_results.keys())
    means = [agent_results[name]["mean_reward"] for name in agent_names_list]
    stds = [agent_results[name]["std_reward"] for name in agent_names_list]
    
    x_pos = np.arange(len(agent_names_list))
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                   color=[colors[i % len(colors)] for i in range(len(agent_names_list))],
                   alpha=0.7)
    
    ax2.axhline(y=195, color='black', linestyle='--', alpha=0.7, label='Solved (195)')
    ax2.set_xlabel('Agent')
    ax2.set_ylabel('Mean Reward ± Std Dev')
    ax2.set_title('Performance Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.replace('_', ' ').title() for name in agent_names_list], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Reward distributions
    for i, (agent_name, results) in enumerate(agent_results.items()):
        episode_rewards = results["episode_rewards"]
        color = colors[i % len(colors)]
        ax3.hist(episode_rewards, bins=15, alpha=0.6, color=color, 
                label=f'{agent_name}', density=True)
    
    ax3.axvline(x=195, color='black', linestyle='--', alpha=0.7, label='Solved (195)')
    ax3.set_xlabel('Episode Reward (Steps)')
    ax3.set_ylabel('Density')
    ax3.set_title('Reward Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Success rates
    success_rates = []
    for agent_name in agent_names_list:
        results = agent_results[agent_name]
        episode_rewards = np.array(results["episode_rewards"])
        success_rate = np.mean(episode_rewards >= 195) * 100
        success_rates.append(success_rate)
    
    bars = ax4.bar(x_pos, success_rates,
                   color=[colors[i % len(colors)] for i in range(len(agent_names_list))],
                   alpha=0.7)
    
    ax4.set_xlabel('Agent')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Success Rate Comparison (≥195 steps)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([name.replace('_', ' ').title() for name in agent_names_list], rotation=45)
    ax4.set_ylim(0, max(100, max(success_rates) * 1.1))
    ax4.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'Agent Comparison: {", ".join([name.replace("_", " ").title() for name in agent_names_list])}',
                fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save comparison plot
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(exist_ok=True)
    comparison_filename = "_vs_".join(agent_names) + "_comparison.png"
    save_path = plots_dir / comparison_filename
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n📈 Performance Summary:")
    print("-" * 60)
    for agent_name in sorted(agent_results.keys(), 
                           key=lambda x: agent_results[x]["mean_reward"], 
                           reverse=True):
        results = agent_results[agent_name]
        episode_rewards = np.array(results["episode_rewards"])
        success_rate = np.mean(episode_rewards >= 195) * 100
        print(f"{agent_name.replace('_', ' ').title():>15}: "
              f"{results['mean_reward']:>6.1f} ± {results['std_reward']:>5.1f} steps "
              f"(success: {success_rate:>5.1f}%)")
    
    print(f"\n✅ Comparison complete! {len(agent_results)} agents analyzed.")


def analyze_random_agent() -> None:
    """Legacy function for backward compatibility."""
    analyze_agent("random")


if __name__ == "__main__":
    # Show available agents and analyze all by default
    available_agents = discover_available_agents()
    
    if not available_agents:
        print("❌ No agent results found!")
        print("Run some agents first: python main.py --agent random --episodes 100")
    else:
        analyze_all_agents()
