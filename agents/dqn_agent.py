"""
Deep Q-Network (DQN) Agent for Cart-Pole

This implements the breakthrough Deep Q-Network algorithm that revolutionized
reinforcement learning by replacing Q-tables with neural networks.

DQN solves the problems we discovered with Q-learning:
1. Curse of Dimensionality: No more 160,000 state discretization
2. Sample Efficiency: Experience replay reuses data
3. Generalization: Similar states share knowledge through function approximation
4. Continuous States: Handle raw state values directly

Learning Goals:
- See how neural networks replace Q-tables
- Understand experience replay and target networks
- Watch dramatic performance improvement (28.5 → 195+ steps)
- Visualize network weights evolving into feature detectors
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from utils.experience_replay import ReplayBuffer


class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture for Cart-Pole.
    
    This neural network replaces the massive 160,000-entry Q-table from
    Q-learning with just ~20,000 parameters that can handle continuous states!
    
    Architecture Design:
    - Input: 4 continuous state values (no discretization!)
    - Hidden: 2 layers of 128 neurons each (ReLU activation)
    - Output: 2 Q-values (one for each action)
    
    Key Innovation: Function approximation generalizes across similar states
    """
    
    def __init__(self, state_size: int = 4, action_size: int = 2, 
                 hidden_sizes: Tuple[int, ...] = (128, 128), seed: Optional[int] = None):
        """
        Initialize the Q-network.
        
        Args:
            state_size: Dimension of input state (4 for Cart-Pole)
            action_size: Number of actions (2 for Cart-Pole)
            hidden_sizes: Sizes of hidden layers
            seed: Random seed for reproducibility
        """
        super(DQNetwork, self).__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor (batch_size x state_size)
            
        Returns:
            Q-values for all actions (batch_size x action_size)
        """
        return self.network(state)
    
    def get_layer_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get current weights for visualization.
        
        Returns:
            Dictionary of layer weights for visualization
        """
        weights = {}
        layer_idx = 0
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                weights[f'layer_{layer_idx}'] = param.data.clone()
                layer_idx += 1
                
        return weights
    
    def get_activations(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate activations for visualization.
        
        Args:
            state: Input state tensor
            
        Returns:
            Dictionary of layer activations
        """
        activations = {}
        x = state
        
        layer_idx = 0
        for i, layer in enumerate(self.network):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations[f'layer_{layer_idx}'] = x.clone()
                layer_idx += 1
        
        # Final output (Q-values)
        activations['q_values'] = x.clone()
        
        return activations


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent that learns to play Cart-Pole using neural networks.
    
    This agent implements the full DQN algorithm with:
    1. Experience replay buffer for sample efficiency
    2. Target network for stable learning targets  
    3. Epsilon-greedy exploration
    4. Neural network function approximation
    
    Expected Performance: Solve Cart-Pole (195+ steps) in 200-300 episodes!
    """
    
    def __init__(self, 
                 state_size: int = 4,
                 action_size: int = 2,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 32,
                 buffer_size: int = 10000,
                 target_update_freq: int = 10,
                 hidden_sizes: Tuple[int, ...] = (128, 128),
                 device: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of input state
            action_size: Number of possible actions
            learning_rate: Neural network learning rate
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Exponential decay rate for epsilon
            epsilon_min: Minimum exploration rate
            batch_size: Mini-batch size for training
            buffer_size: Replay buffer capacity
            target_update_freq: Episodes between target network updates
            hidden_sizes: Hidden layer dimensions
            device: Device to run on ('cpu', 'cuda', or None for auto)
            seed: Random seed for reproducibility
        """
        # Create a dummy action space for BaseAgent compatibility
        import gymnasium as gym
        action_space = gym.spaces.Discrete(action_size)
        super().__init__(action_space)
        
        # Override the name from BaseAgent
        self.name = "dqn"
        
        # Set random seeds for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Store hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.episodes_trained = 0
        
        # Device setup with intelligent selection
        if device is None:
            # Smart device selection
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                device_desc = "NVIDIA CUDA GPU"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # For small networks like DQN, CPU is often faster than MPS on Apple Silicon
                # But we'll let users opt-in to MPS if they want to experiment
                self.device = torch.device("cpu")  # Default to CPU for better performance
                device_desc = f"CPU (MPS available but CPU faster for small networks)"
            else:
                self.device = torch.device("cpu")
                device_desc = "CPU (no GPU acceleration available)"
        else:
            self.device = torch.device(device)
            device_desc = f"User-specified: {device}"
        
        print(f"🧠 DQN Agent using device: {self.device}")
        print(f"   💡 {device_desc}")
        
        # Neural Networks
        self.q_network = DQNetwork(state_size, action_size, hidden_sizes, seed).to(self.device)
        self.target_network = DQNetwork(state_size, action_size, hidden_sizes, seed).to(self.device)
        
        # Initialize target network with same weights as main network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size, seed)
        
        # Training statistics
        self.training_stats = {
            'losses': [],
            'q_values': [],
            'epsilon_history': [],
            'episode_rewards': []
        }
        
        print(f"📊 Network Architecture:")
        print(f"   Input: {state_size} → Hidden: {hidden_sizes} → Output: {action_size}")
        print(f"   Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"   vs Q-learning table: 160,000 entries (87% reduction!)")
    
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy with neural network Q-values.
        
        Args:
            state: Current environment state
            
        Returns:
            Selected action (0 or 1)
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values from network (no gradient needed for action selection)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Select action with highest Q-value
        return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self) -> Optional[float]:
        """
        Train the neural network using a mini-batch of experiences.
        
        This is where the magic happens - we sample random experiences
        from the replay buffer and update the network weights.
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        # Can't train without enough experiences
        if not self.memory.can_sample(self.batch_size):
            return None
        
        # Sample random mini-batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values for taken actions
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values using target network (for stability)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss (Mean Squared Error between predicted and target Q-values)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Store training statistics
        with torch.no_grad():
            avg_q_value = current_q_values.mean().item()
            self.training_stats['losses'].append(loss.item())
            self.training_stats['q_values'].append(avg_q_value)
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """
        Copy weights from main network to target network.
        
        This prevents the "moving targets" problem where Q-value targets
        change as we update the network, causing instability.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_stats['epsilon_history'].append(self.epsilon)
    
    def episode_finished(self, episode_reward: float) -> None:
        """
        Called at the end of each episode for bookkeeping.
        
        Args:
            episode_reward: Total reward for the episode
        """
        self.episodes_trained += 1
        self.training_stats['episode_rewards'].append(episode_reward)
        
        # Update target network periodically
        if self.episodes_trained % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay exploration
        self.decay_epsilon()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        buffer_stats = self.memory.get_buffer_stats()
        
        return {
            'episodes_trained': self.episodes_trained,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'buffer_utilization': buffer_stats.get('utilization', 0),
            'avg_loss': np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0,
            'avg_q_value': np.mean(self.training_stats['q_values'][-100:]) if self.training_stats['q_values'] else 0,
            'recent_rewards': self.training_stats['episode_rewards'][-10:] if self.training_stats['episode_rewards'] else []
        }

    def get_info(self) -> dict[str, Any]:
        """Return agent information for evaluation compatibility."""
        return {
            "name": "DQN Agent",
            "type": "dqn",
            "episodes_trained": self.episodes_trained,
            "epsilon": self.epsilon,
            "device": str(self.device)
        }
    
    def save(self, file_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            file_path: Path to save the model
        """
        save_dict = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'training_stats': self.training_stats,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }
        
        torch.save(save_dict, file_path)
        print(f"💾 DQN model saved to {file_path}")
    
    def load(self, file_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            file_path: Path to the saved model
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        save_dict = torch.load(file_path, map_location=self.device)
        
        self.q_network.load_state_dict(save_dict['q_network_state_dict'])
        self.target_network.load_state_dict(save_dict['target_network_state_dict'])
        self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        
        self.epsilon = save_dict['epsilon']
        self.episodes_trained = save_dict['episodes_trained']
        self.training_stats = save_dict['training_stats']
        
        print(f"📁 DQN model loaded from {file_path}")
        print(f"📊 Episodes trained: {self.episodes_trained}, Current ε: {self.epsilon:.3f}")


def train_dqn_agent(episodes: int = 500, render: bool = False, 
                    save_model: bool = True, verbose: bool = True,
                    visualize: bool = False, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Train a DQN agent on Cart-Pole environment.
    
    Args:
        episodes: Number of episodes to train
        render: Whether to render the environment
        save_model: Whether to save the trained model
        verbose: Whether to print training progress
        visualize: Whether to show real-time neural network visualization
        device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
        
    Returns:
        Dictionary with training results and statistics
    """
    import gymnasium as gym
    
    print("🧠 Training Deep Q-Network (DQN) Agent...")
    print("=" * 60)
    
    # Create environment and agent
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    agent = DQNAgent(seed=42, device=device)
    
    # Initialize visualization dashboard if requested
    dashboard = None
    if visualize:
        try:
            from visualization.neural_network_visualizer import TrainingDashboard
            dashboard = TrainingDashboard(agent, update_frequency=20)
            print("🎨 Real-time visualization dashboard activated!")
        except ImportError:
            print("⚠️  Visualization not available - continuing without it")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    
    # Sample state for visualization
    sample_state = np.array([0.1, -0.5, 0.02, 0.3])  # Slightly off-center Cart-Pole state
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            step_count = 0
            episode_losses = []
            
            while True:
                # Agent selects action
                action = agent.select_action(state)
                
                # Environment step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train if enough experiences collected
                loss = agent.train()
                if loss is not None:
                    episode_losses.append(loss)
                
                # Update state and counters
                state = next_state
                total_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # Episode finished
            agent.episode_finished(total_reward)
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            if episode_losses:
                training_losses.append(np.mean(episode_losses))
            
            # Update visualization dashboard
            if dashboard:
                dashboard.update_dashboard(episode + 1, total_reward, sample_state)
            
            # Progress reporting
            if verbose and (episode + 1) % 50 == 0:
                recent_rewards = episode_rewards[-50:]
                stats = agent.get_training_stats()
                
                print(f"Episode {episode + 1:3d}/{episodes}: "
                      f"Reward: {total_reward:3.0f} | "
                      f"Avg: {np.mean(recent_rewards):5.1f} | "
                      f"ε: {stats['epsilon']:.3f} | "
                      f"Loss: {stats['avg_loss']:.4f} | "
                      f"Q-val: {stats['avg_q_value']:.2f}")
                
                # Check if solved (195+ average over 100 episodes)
                if len(episode_rewards) >= 100:
                    avg_100 = np.mean(episode_rewards[-100:])
                    if avg_100 >= 195:
                        print(f"🎉 SOLVED! Average reward over 100 episodes: {avg_100:.1f}")
                        break
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    finally:
        env.close()
        
        # Save final visualization
        if dashboard:
            dashboard.save_dashboard()
            print("💾 Final visualizations saved!")
    
    # Calculate final statistics
    final_stats = {
        'agent_name': 'DQN',
        'episodes_trained': len(episode_rewards),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_losses': training_losses,
        'mean_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.std(episode_rewards),
        'max_reward': np.max(episode_rewards) if episode_rewards else 0,
        'solved': np.mean(episode_rewards[-100:]) >= 195 if len(episode_rewards) >= 100 else False,
        'agent_stats': agent.get_training_stats()
    }
    
    # Save results
    results_dir = Path("outputs/results")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    import json
    # Convert numpy types and booleans to JSON serializable types
    serializable_stats = {}
    for k, v in final_stats.items():
        if k == 'agent_stats':
            continue
        elif k == 'solved':
            serializable_stats[k] = bool(v)  # Ensure it's a Python bool
        elif isinstance(v, np.ndarray):
            serializable_stats[k] = v.tolist()
        elif isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
            serializable_stats[k] = v.item()
        else:
            serializable_stats[k] = v
    
    # Add proper agent schema wrapper for visualization compatibility
    final_results = {
        "agent": {"name": "DQN Agent", "type": "dqn"},
        **serializable_stats
    }
    
    with open(results_dir / "dqn_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Save trained model
    if save_model:
        models_dir = Path("outputs/models")
        models_dir.mkdir(exist_ok=True, parents=True)
        agent.save(str(models_dir / "dqn_agent.pth"))
    
    # Print final results
    if verbose:
        print(f"\n🎯 DQN Training Complete!")
        print(f"Episodes: {final_stats['episodes_trained']}")
        print(f"Final performance: {final_stats['mean_reward']:.1f} ± {final_stats['std_reward']:.1f} steps")
        print(f"Max episode: {final_stats['max_reward']:.0f} steps")
        print(f"Solved: {'✅ Yes' if final_stats['solved'] else '❌ Not yet'}")
        
        from utils.constants import format_baseline_comparison, calculate_improvement, CART_POLE_SUCCESS_THRESHOLD
        
        print(format_baseline_comparison('dqn', final_stats['mean_reward'], final_stats['std_reward']))
        
        if final_stats['mean_reward'] > CART_POLE_SUCCESS_THRESHOLD:
            improvement = calculate_improvement(final_stats['mean_reward'], 'random')
            print(f"   🚀 Improvement over random: {improvement:+.0f}%")
    
    return final_stats


def demo_trained_dqn(model_path: str = "outputs/models/dqn_agent.pth", 
                     episodes: int = 5, render: bool = True) -> None:
    """
    Demo a trained DQN model playing Cart-Pole.
    
    Args:
        model_path: Path to the saved DQN model
        episodes: Number of episodes to play
        render: Whether to show the game visually
    """
    import gymnasium as gym
    from pathlib import Path
    
    print(f"🎮 DQN Demo: Loading trained model from {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("💡 Train a model first with: python main.py --agent dqn --episodes 1000")
        return
    
    # Create environment and agent
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    agent = DQNAgent(seed=42)
    
    try:
        # Load the trained model
        agent.load(model_path)
        agent.epsilon = 0.0  # No exploration - pure exploitation
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model was trained for {agent.episodes_trained} episodes")
        print(f"🎯 Running {episodes} demo episodes...")
        
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            step_count = 0
            
            print(f"\n🎮 Episode {episode + 1}:")
            
            while True:
                # Agent selects action (no exploration)
                action = agent.select_action(state)
                
                # Environment step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                step_count += 1
                state = next_state
                
                if render:
                    import time
                    time.sleep(0.05)  # Slow down for better viewing
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            print(f"   Steps survived: {step_count}")
            print(f"   Total reward: {total_reward}")
            
            if total_reward >= 195:
                print("   🎉 Episode solved (≥195 steps)!")
            
        # Summary
        avg_reward = np.mean(episode_rewards)
        print(f"\n📊 Demo Summary:")
        print(f"   Average performance: {avg_reward:.1f} steps")
        print(f"   Success rate: {sum(1 for r in episode_rewards if r >= 195)}/{episodes}")
        print(f"   Best episode: {max(episode_rewards):.0f} steps")
        
        if avg_reward >= 195:
            print("🎉 Trained DQN consistently solves Cart-Pole!")
        else:
            print("🎯 Model shows good performance but not consistently solving")
    
    finally:
        env.close()


if __name__ == "__main__":
    # Quick test of network architecture
    print("🧪 Testing DQN Network Architecture...")
    
    # Create a test network
    network = DQNetwork(state_size=4, action_size=2, hidden_sizes=(128, 128))
    
    # Test forward pass
    test_state = torch.randn(1, 4)  # Batch size 1, state size 4
    q_values = network(test_state)
    
    print(f"✅ Network test successful!")
    print(f"   Input shape: {test_state.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values: {q_values.squeeze().tolist()}")
    print(f"   Parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    print(f"\n🎯 Ready for training with DQN!")