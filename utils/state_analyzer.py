"""
State Analysis Tool for Cart-Pole Environment

This module collects and analyzes state distributions during episodes to help
understand the environment dynamics before implementing Q-learning.

Learning Goals:
- Understand the continuous state space structure
- See correlations between states and optimal actions  
- Identify state regions that matter most for control
- Inform discretization strategy for Q-learning
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from agents.base_agent import BaseAgent


class StateCollector:
    """
    Collects state data during agent episodes for analysis.
    
    This helps us understand the environment before implementing Q-learning
    by showing us what the continuous state space actually looks like.
    """
    
    def __init__(self):
        """Initialize the state collector."""
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[bool] = []
        
        # For episode tracking
        self.episode_states: list[list[np.ndarray]] = []
        self.episode_actions: list[list[int]] = []
        self.episode_rewards: list[list[float]] = []
        self.episode_lengths: list[int] = []
    
    def add_transition(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool) -> None:
        """
        Add a single transition to the collection.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state.copy())
        self.dones.append(done)
    
    def start_episode(self) -> None:
        """Start collecting data for a new episode."""
        self.episode_states.append([])
        self.episode_actions.append([])
        self.episode_rewards.append([])
    
    def add_episode_step(self, state: np.ndarray, action: int, reward: float) -> None:
        """
        Add a step to the current episode.
        
        Args:
            state: Current state
            action: Action taken  
            reward: Reward received
        """
        if not self.episode_states:
            self.start_episode()
            
        self.episode_states[-1].append(state.copy())
        self.episode_actions[-1].append(action)
        self.episode_rewards[-1].append(reward)
    
    def end_episode(self, final_state: np.ndarray = None) -> None:
        """
        Finish the current episode.
        
        Args:
            final_state: Final state of the episode (optional)
        """
        if self.episode_states and self.episode_states[-1]:
            if final_state is not None:
                self.episode_states[-1].append(final_state.copy())
            self.episode_lengths.append(len(self.episode_rewards[-1]))
    
    def get_state_statistics(self) -> dict[str, Any]:
        """
        Calculate statistics about the collected states.
        
        Returns:
            Dictionary with state statistics
        """
        if not self.states:
            return {}
            
        states_array = np.array(self.states)
        
        return {
            "total_transitions": len(self.states),
            "total_episodes": len(self.episode_lengths),
            "state_means": np.mean(states_array, axis=0),
            "state_stds": np.std(states_array, axis=0),
            "state_mins": np.min(states_array, axis=0),
            "state_maxs": np.max(states_array, axis=0),
            "avg_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "action_distribution": np.bincount(self.actions) / len(self.actions) if self.actions else np.array([])
        }


def collect_state_data(agent: BaseAgent, episodes: int = 100) -> StateCollector:
    """
    Collect state data by running an agent for multiple episodes.
    
    Args:
        agent: Agent to run
        episodes: Number of episodes to collect
        
    Returns:
        StateCollector with all the collected data
    """
    print(f"🔬 Collecting state data from {agent.name} agent over {episodes} episodes...")
    
    env = gym.make("CartPole-v1")
    collector = StateCollector()
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            collector.start_episode()
            
            done = False
            while not done:
                action = agent.select_action(state)
                collector.add_episode_step(state, action, 1.0)  # CartPole gives +1 per step
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Add to transition collection
                collector.add_transition(state, action, reward, next_state, done)
                
                state = next_state
            
            collector.end_episode(state)
            
            if (episode + 1) % 20 == 0:
                print(f"📊 Collected {episode + 1}/{episodes} episodes...")
    
    finally:
        env.close()
    
    print(f"✅ Data collection complete! {len(collector.states)} transitions collected.")
    return collector


def plot_state_distributions(collector: StateCollector, save_path: str = None) -> None:
    """
    Plot histograms of each state component to understand the state space.
    
    Args:
        collector: StateCollector with data
        save_path: Optional path to save the plot
    """
    if not collector.states:
        print("❌ No state data to plot!")
        return
    
    states_array = np.array(collector.states)
    state_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    state_limits = [
        (-2.4, 2.4),      # Cart position limits
        (-3.0, 3.0),      # Velocity (estimated)
        (-0.21, 0.21),    # Angle limits (±12 degrees in radians)
        (-3.0, 3.0)       # Angular velocity (estimated)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    stats = collector.get_state_statistics()
    
    for i, (ax, name, limits) in enumerate(zip(axes, state_names, state_limits)):
        state_values = states_array[:, i]
        
        # Plot histogram
        ax.hist(state_values, bins=50, alpha=0.7, color='steelblue', density=True)
        
        # Add statistics lines
        mean_val = stats["state_means"][i]
        std_val = stats["state_stds"][i]
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7,
                  label=f'±1 Std: {std_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
        
        # Add environment limits
        ax.axvline(limits[0], color='black', linestyle='-', alpha=0.5, 
                  label=f'Env limits: [{limits[0]:.1f}, {limits[1]:.1f}]')
        ax.axvline(limits[1], color='black', linestyle='-', alpha=0.5)
        
        ax.set_xlabel(f'{name} Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Distribution\n(n={len(state_values):,} transitions)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'State Space Analysis\nAgent: {collector.states[0].__class__.__name__ if hasattr(collector.states[0], "__class__") else "Unknown"} | '
                 f'Episodes: {stats["total_episodes"]} | '
                 f'Avg Length: {stats["avg_episode_length"]:.1f} steps',
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 State distribution plot saved to {save_path}")
    
    plt.show()


def plot_action_correlations(collector: StateCollector, save_path: str = None) -> None:
    """
    Plot correlations between states and actions taken.
    
    Args:
        collector: StateCollector with data
        save_path: Optional path to save the plot
    """
    if not collector.states or not collector.actions:
        print("❌ No state-action data to analyze!")
        return
    
    states_array = np.array(collector.states)
    actions_array = np.array(collector.actions)
    
    state_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        state_values = states_array[:, i]
        
        # Separate states by action taken
        left_actions = state_values[actions_array == 0]  # Action 0 = Left
        right_actions = state_values[actions_array == 1] # Action 1 = Right
        
        # Plot distributions for each action
        ax.hist(left_actions, bins=30, alpha=0.6, color='red', 
               label=f'Left Action (n={len(left_actions):,})', density=True)
        ax.hist(right_actions, bins=30, alpha=0.6, color='blue',
               label=f'Right Action (n={len(right_actions):,})', density=True)
        
        # Add vertical line at zero for reference
        if i in [0, 2]:  # Position and angle
            ax.axvline(0, color='black', linestyle='-', alpha=0.3, label='Zero')
        
        ax.set_xlabel(f'{name} Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} vs Action Choice')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    stats = collector.get_state_statistics()
    action_dist = stats["action_distribution"]
    
    plt.suptitle(f'State-Action Correlations\n'
                 f'Episodes: {stats["total_episodes"]} | '
                 f'Action Split: Left {action_dist[0]:.1%} / Right {action_dist[1]:.1%}',
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 State-action correlation plot saved to {save_path}")
    
    plt.show()


def suggest_discretization_bins(collector: StateCollector, target_bins: int = 20) -> dict[str, Any]:
    """
    Analyze state distributions to suggest good discretization bins for Q-learning.
    
    Args:
        collector: StateCollector with data
        target_bins: Target number of bins per dimension
        
    Returns:
        Dictionary with discretization suggestions
    """
    if not collector.states:
        print("❌ No state data to analyze!")
        return {}
    
    states_array = np.array(collector.states)
    stats = collector.get_state_statistics()
    
    state_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    
    # Calculate discretization ranges based on actual data
    discretization_info = {}
    
    for i, name in enumerate(state_names):
        values = states_array[:, i]
        mean_val = stats["state_means"][i]
        std_val = stats["state_stds"][i]
        
        # Use mean ± 2*std to cover ~95% of data
        data_min = mean_val - 2 * std_val
        data_max = mean_val + 2 * std_val
        
        # But also consider actual min/max
        actual_min = stats["state_mins"][i] 
        actual_max = stats["state_maxs"][i]
        
        # Use the wider range
        final_min = min(data_min, actual_min)
        final_max = max(data_max, actual_max)
        
        # Calculate bin edges
        bin_edges = np.linspace(final_min, final_max, target_bins + 1)
        bin_width = (final_max - final_min) / target_bins
        
        discretization_info[f"state_{i}"] = {
            "name": name,
            "num_bins": target_bins,
            "range": (final_min, final_max),
            "bin_width": bin_width,
            "bin_edges": bin_edges,
            "data_coverage": np.mean((values >= final_min) & (values <= final_max))
        }
    
    # Calculate total Q-table size
    total_states = target_bins ** 4  # 4 state dimensions
    total_entries = total_states * 2  # 2 actions
    
    discretization_info["summary"] = {
        "bins_per_dimension": target_bins,
        "total_discrete_states": total_states,
        "q_table_size": total_entries,
        "memory_estimate_mb": total_entries * 8 / 1024 / 1024  # 8 bytes per float64
    }
    
    return discretization_info


def analyze_environment(agent: BaseAgent, episodes: int = 100) -> None:
    """
    Complete environment analysis workflow.
    
    Args:
        agent: Agent to analyze with
        episodes: Number of episodes to run
    """
    print(f"🔬 Starting environment analysis with {agent.name} agent...")
    print(f"📊 Running {episodes} episodes to collect state data...\n")
    
    # Collect data
    collector = collect_state_data(agent, episodes)
    
    # Create output directory
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate visualizations
    print("\n📈 Creating state distribution plots...")
    plot_state_distributions(
        collector, 
        save_path=str(plots_dir / f"state_distributions_{agent.name}.png")
    )
    
    print("\n🔗 Creating state-action correlation plots...")  
    plot_action_correlations(
        collector,
        save_path=str(plots_dir / f"state_action_correlations_{agent.name}.png")
    )
    
    # Print statistics
    stats = collector.get_state_statistics()
    print(f"\n📊 Environment Analysis Summary:")
    print("=" * 60)
    print(f"Total transitions collected: {stats['total_transitions']:,}")
    print(f"Total episodes: {stats['total_episodes']:,}")
    print(f"Average episode length: {stats['avg_episode_length']:.1f} steps")
    print(f"Action distribution: Left {stats['action_distribution'][0]:.1%} / Right {stats['action_distribution'][1]:.1%}")
    
    print(f"\n🏠 State Space Characteristics:")
    state_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    for i, name in enumerate(state_names):
        print(f"{name:>20}: {stats['state_means'][i]:>8.3f} ± {stats['state_stds'][i]:>6.3f} "
              f"[{stats['state_mins'][i]:>7.3f}, {stats['state_maxs'][i]:>7.3f}]")
    
    # Suggest discretization
    print(f"\n🔢 Q-Learning Discretization Suggestions:")
    print("-" * 60)
    disc_info = suggest_discretization_bins(collector, target_bins=20)
    
    if disc_info:
        summary = disc_info["summary"]
        print(f"Recommended bins per dimension: {summary['bins_per_dimension']}")
        print(f"Total discrete states: {summary['total_discrete_states']:,}")
        print(f"Q-table size: {summary['q_table_size']:,} entries")
        print(f"Estimated memory usage: {summary['memory_estimate_mb']:.1f} MB")
        
        print(f"\nPer-dimension discretization ranges:")
        for key, info in disc_info.items():
            if key.startswith("state_"):
                coverage = info["data_coverage"]
                print(f"{info['name']:>20}: [{info['range'][0]:>7.3f}, {info['range'][1]:>7.3f}] "
                      f"(width: {info['bin_width']:.4f}, coverage: {coverage:.1%})")
    
    print(f"\n✅ Environment analysis complete!")
    print(f"📁 Plots saved in {plots_dir}/")
    print(f"\n💡 Next Steps:")
    print("   1. Review state distributions to understand environment dynamics")
    print("   2. Check state-action correlations for patterns")
    print("   3. Use discretization suggestions for Q-learning implementation")
    print("   4. Consider if 20 bins per dimension gives good coverage vs. manageable table size")


if __name__ == "__main__":
    # Example usage with random agent
    from agents.random_agent import RandomAgent
    
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env.action_space)
    env.close()
    
    analyze_environment(agent, episodes=100)