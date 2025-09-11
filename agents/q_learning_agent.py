"""
Q-Learning Agent for Cart-Pole Environment

This implements tabular Q-learning with state discretization, representing the first 
"real" reinforcement learning algorithm in our learning journey.

Key Concepts Demonstrated:
- State discretization (continuous → discrete)
- Q-table (state-action value function)
- Bellman equation updates
- Epsilon-greedy exploration
- Temporal difference learning

Learning Goals:
- Understand how Q-values represent expected future rewards
- See exploration vs exploitation trade-off in action
- Experience the curse of dimensionality with state spaces
- Learn why discretization choices matter for performance
"""

from pathlib import Path
from typing import Any, Tuple

import gymnasium as gym
import json
import numpy as np
import pickle

from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent with state discretization.
    
    This agent learns by maintaining a table Q(s,a) that estimates the expected
    future reward for taking action 'a' in state 's'. The famous Q-learning
    update rule is:
    
    Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    
    The challenge in Cart-Pole is that states are continuous, so we must
    discretize them into bins to create a finite Q-table.
    """
    
    def __init__(self, 
                 action_space: gym.Space,
                 state_ranges: list[Tuple[float, float]] = None,
                 n_bins: int = 20,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize the Q-Learning agent.
        
        Args:
            action_space: Environment's action space
            state_ranges: List of (min, max) for each state dimension
            n_bins: Number of bins per state dimension
            learning_rate: Learning rate (α) for Q-updates
            discount_factor: Discount factor (γ) for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay factor for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """
        super().__init__(action_space)
        self.name = "Q-Learning"
        
        # Q-learning hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon
        
        # State discretization setup
        self.n_bins = n_bins
        
        # Default ranges based on our state analysis (from analyzer output)
        if state_ranges is None:
            self.state_ranges = [
                (-0.25, 0.25),     # Cart position (slightly wider than observed)
                (-2.0, 2.0),       # Cart velocity (conservative estimate)
                (-0.25, 0.25),     # Pole angle (slightly wider than ±12°)
                (-3.0, 3.0)        # Pole angular velocity (conservative)
            ]
        else:
            self.state_ranges = state_ranges
        
        # Create bin edges for each dimension
        self.bin_edges = []
        for min_val, max_val in self.state_ranges:
            edges = np.linspace(min_val, max_val, n_bins + 1)
            self.bin_edges.append(edges)
        
        # Initialize Q-table: [state_bins^4, n_actions]
        self.n_actions = action_space.n
        q_shape = tuple([n_bins] * 4 + [self.n_actions])
        self.q_table = np.zeros(q_shape)
        
        # Training statistics
        self.episode_count = 0
        self.training_rewards = []
        self.training_epsilons = []
        self.q_value_history = []  # Track Q-value evolution
        
        print(f"🧮 Q-Learning Agent Initialized:")
        print(f"   State bins per dimension: {n_bins}")
        print(f"   Total discrete states: {n_bins**4:,}")
        print(f"   Q-table size: {self.q_table.size:,} entries")
        print(f"   Memory usage: ~{self.q_table.nbytes / 1024 / 1024:.1f} MB")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Discount factor: {discount_factor}")
        print(f"   Epsilon: {epsilon} → {epsilon_min} (decay: {epsilon_decay})")

    def discretize_state(self, state: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Convert continuous state to discrete bin indices.
        
        This is the key function that bridges continuous and discrete worlds!
        Each continuous state component gets mapped to a bin index.
        
        Args:
            state: Continuous state [position, velocity, angle, angular_velocity]
            
        Returns:
            Tuple of bin indices for each state dimension
        """
        discrete_state = []
        
        for i, value in enumerate(state):
            # Clip value to be within our defined ranges
            clipped_value = np.clip(value, self.state_ranges[i][0], self.state_ranges[i][1])
            
            # Find which bin this value belongs to
            bin_index = np.digitize(clipped_value, self.bin_edges[i]) - 1
            
            # Ensure bin index is within valid range
            bin_index = np.clip(bin_index, 0, self.n_bins - 1)
            discrete_state.append(bin_index)
        
        return tuple(discrete_state)
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon: explore (random action)
        With probability 1-epsilon: exploit (greedy action from Q-table)
        
        Args:
            state: Current environment state
            
        Returns:
            Action to take (0=left, 1=right)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return self.action_space.sample()
        else:
            # Exploit: greedy action based on Q-values
            discrete_state = self.discretize_state(state)
            q_values = self.q_table[discrete_state]
            return np.argmax(q_values)
    
    def update_q_value(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool) -> None:
        """
        Update Q-table using the Bellman equation.
        
        This is the heart of Q-learning! The update rule is:
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Breaking it down:
        - r + γ max Q(s',a'): Target value (immediate reward + discounted future)
        - Q(s,a): Current estimate
        - α: Learning rate (how much to update)
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[discrete_state][action]
        
        if done:
            # Terminal state: no future rewards
            target = reward
        else:
            # Non-terminal: reward + discounted max future Q-value
            max_next_q = np.max(self.q_table[discrete_next_state])
            target = reward + self.discount_factor * max_next_q
        
        # Q-learning update (Bellman equation)
        td_error = target - current_q
        new_q = current_q + self.learning_rate * td_error
        self.q_table[discrete_state][action] = new_q
    
    def train_episode(self, env: gym.Env) -> float:
        """
        Train the agent for one episode.
        
        Returns:
            Total reward earned in the episode
        """
        state, _ = env.reset()
        total_reward = 0.0
        
        while True:
            # Select and execute action
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update Q-table
            self.update_q_value(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return total_reward
    
    def train(self, env: gym.Env, episodes: int, verbose: bool = True) -> dict[str, Any]:
        """
        Train the agent for multiple episodes.
        
        Args:
            env: Environment to train in
            episodes: Number of episodes to train
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        if verbose:
            print(f"🎯 Starting Q-Learning training for {episodes} episodes...")
            print(f"🔍 Initial exploration rate: {self.epsilon:.3f}")
        
        for episode in range(episodes):
            # Train one episode
            episode_reward = self.train_episode(env)
            self.training_rewards.append(episode_reward)
            self.training_epsilons.append(self.epsilon)
            
            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            
            # Track Q-value statistics periodically
            if episode % 50 == 0:
                q_stats = self.get_q_value_stats()
                self.q_value_history.append({
                    'episode': episode,
                    'mean_q': q_stats['mean_q'],
                    'max_q': q_stats['max_q'],
                    'min_q': q_stats['min_q'],
                    'nonzero_q': q_stats['nonzero_entries']
                })
            
            # Progress reporting
            if verbose and (episode + 1) % 100 == 0:
                recent_avg = np.mean(self.training_rewards[-100:])
                print(f"Episode {episode + 1:4d}: "
                      f"Avg reward: {recent_avg:6.1f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Last reward: {episode_reward:3.0f}")
        
        self.episode_count += episodes
        
        if verbose:
            final_avg = np.mean(self.training_rewards[-100:])
            print(f"\n✅ Training complete!")
            print(f"📊 Final 100-episode average: {final_avg:.1f} steps")
            print(f"🔍 Final exploration rate: {self.epsilon:.3f}")
        
        return {
            'episodes_trained': episodes,
            'final_avg_reward': np.mean(self.training_rewards[-min(100, len(self.training_rewards)):]),
            'training_rewards': self.training_rewards.copy(),
            'training_epsilons': self.training_epsilons.copy(),
            'q_value_history': self.q_value_history.copy()
        }
    
    def get_q_value_stats(self) -> dict[str, float]:
        """Get statistics about the current Q-table."""
        flat_q = self.q_table.flatten()
        nonzero_q = flat_q[flat_q != 0]
        
        return {
            'mean_q': np.mean(flat_q),
            'std_q': np.std(flat_q),
            'max_q': np.max(flat_q),
            'min_q': np.min(flat_q),
            'nonzero_entries': len(nonzero_q),
            'total_entries': len(flat_q),
            'exploration_ratio': len(nonzero_q) / len(flat_q)
        }
    
    def evaluate(self, env: gym.Env, episodes: int = 100) -> dict[str, Any]:
        """
        Evaluate the trained agent (no learning, no exploration).
        
        Args:
            env: Environment to evaluate in
            episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation results
        """
        # Save current epsilon and set to 0 (no exploration)
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0.0
            
            while True:
                action = self.select_action(state)  # Greedy action only
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
        
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        return {
            'agent': self.get_info(),
            'episodes': episodes,
            'episode_rewards': episode_rewards,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'success_rate': np.mean(np.array(episode_rewards) >= 195),
            'q_stats': self.get_q_value_stats()
        }
    
    def get_info(self) -> dict[str, Any]:
        """Get agent information for logging."""
        return {
            'name': self.name,
            'type': 'tabular_q_learning',
            'n_bins': self.n_bins,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_initial': self.initial_epsilon,
            'epsilon_current': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'episodes_trained': self.episode_count,
            'state_ranges': self.state_ranges,
            'q_table_size': self.q_table.size
        }
    
    def save(self, filepath: str) -> None:
        """Save the trained agent."""
        save_data = {
            'q_table': self.q_table,
            'state_ranges': self.state_ranges,
            'n_bins': self.n_bins,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'initial_epsilon': self.initial_epsilon,
            'episode_count': self.episode_count,
            'training_rewards': self.training_rewards,
            'training_epsilons': self.training_epsilons,
            'q_value_history': self.q_value_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"💾 Q-Learning agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load a trained agent."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = save_data['q_table']
        self.state_ranges = save_data['state_ranges']
        self.n_bins = save_data['n_bins']
        self.learning_rate = save_data['learning_rate']
        self.discount_factor = save_data['discount_factor']
        self.epsilon = save_data['epsilon']
        self.epsilon_decay = save_data['epsilon_decay']
        self.epsilon_min = save_data['epsilon_min']
        self.initial_epsilon = save_data['initial_epsilon']
        self.episode_count = save_data['episode_count']
        self.training_rewards = save_data.get('training_rewards', [])
        self.training_epsilons = save_data.get('training_epsilons', [])
        self.q_value_history = save_data.get('q_value_history', [])
        
        # Recreate bin edges
        self.bin_edges = []
        for min_val, max_val in self.state_ranges:
            edges = np.linspace(min_val, max_val, self.n_bins + 1)
            self.bin_edges.append(edges)
        
        print(f"📂 Q-Learning agent loaded from {filepath}")


def train_q_learning_agent(episodes: int = 1000, render: bool = False) -> dict[str, Any]:
    """
    Train a Q-learning agent and return results.
    
    Args:
        episodes: Number of training episodes
        render: Whether to render during training (slow)
        
    Returns:
        Training and evaluation results
    """
    print("🧮 Training Q-Learning Agent...")
    print("=" * 50)
    
    # Create environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    # Create agent with state ranges from our analyzer
    agent = QLearningAgent(
        action_space=env.action_space,
        state_ranges=[
            (-0.25, 0.25),     # Cart position
            (-2.0, 2.0),       # Cart velocity  
            (-0.25, 0.25),     # Pole angle
            (-3.0, 3.0)        # Pole angular velocity
        ],
        n_bins=20,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    try:
        # Train the agent
        training_results = agent.train(env, episodes=episodes, verbose=True)
        
        print(f"\n📊 Evaluating trained agent...")
        eval_results = agent.evaluate(env, episodes=100)
        
        print(f"\n🎯 Q-Learning Results:")
        print(f"   Training episodes: {episodes}")
        print(f"   Evaluation average: {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f} steps")
        print(f"   Success rate: {eval_results['success_rate']:.1%} (≥195 steps)")
        print(f"   Q-table exploration: {eval_results['q_stats']['exploration_ratio']:.1%}")
        
        # Save agent and results
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Save agent
        agent.save(str(outputs_dir / "q_learning_agent.pkl"))
        
        # Save evaluation results
        results_dir = outputs_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "q_learning_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Combine training and evaluation results
        final_results = {
            **eval_results,
            'training_stats': training_results
        }
        
        return final_results
        
    finally:
        env.close()


if __name__ == "__main__":
    # Train and evaluate Q-learning agent
    results = train_q_learning_agent(episodes=1000, render=False)
    
    print(f"\n✅ Q-Learning training complete!")
    print(f"📁 Results saved in outputs/")
    print(f"📊 Run 'python main.py --visualize q_learning' to see performance plots")