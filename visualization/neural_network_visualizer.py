"""
Real-Time Neural Network Visualizer for DQN Training

This module creates an interactive dashboard that shows how the neural network
evolves during training - one of the most fascinating aspects of deep RL!

What You'll See:
1. Network Architecture: Nodes changing color/size based on weights
2. Weight Heatmaps: Layer-by-layer weight matrices evolving
3. Training Metrics: Loss, Q-values, rewards in real-time
4. Q-Value Surface: How the value function landscape changes

Learning Goals:
- Watch neural networks "discover" Cart-Pole physics
- See weights organize into feature detectors
- Understand how different layers specialize
- Experience the "aha moments" when networks suddenly improve
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import torch
import torch.nn as nn

from agents.dqn_agent import DQNAgent, DQNetwork


class NetworkArchitectureVisualizer:
    """
    Visualizes the neural network as an interactive graph with nodes and connections.
    
    Node colors represent activation strength, connection thickness shows weights.
    This gives an intuitive view of how information flows through the network.
    """
    
    def __init__(self, network: DQNetwork, figsize: Tuple[int, int] = (14, 10)):
        """
        Initialize the network visualizer.
        
        Args:
            network: The DQN network to visualize
            figsize: Figure size for the plot
        """
        self.network = network
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 8)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title('🧠 DQN Neural Network Architecture\nNodes: Activation Strength | Lines: Weight Magnitude', 
                         fontsize=16, fontweight='bold')
        
        # Make figure more prominent
        self.fig.patch.set_facecolor('white')
        print("🧠 NetworkArchitectureVisualizer initialized successfully")
        
        # Network structure
        self.layer_sizes = [4] + list(network.hidden_sizes) + [2]  # [4, 128, 128, 2]
        self.layer_positions = self._calculate_layer_positions()
        self.node_positions = self._calculate_node_positions()
        
        # Visual elements (will be created in setup)
        self.nodes = []
        self.connections = []
        self.weight_text = None
        
        self.setup_visualization()
    
    def _calculate_layer_positions(self) -> List[float]:
        """Calculate x-positions for each layer."""
        return np.linspace(1, 9, len(self.layer_sizes)).tolist()
    
    def _calculate_node_positions(self) -> Dict[str, List[Tuple[float, float]]]:
        """Calculate (x, y) positions for each node in each layer."""
        positions = {}
        
        for layer_idx, (layer_size, x_pos) in enumerate(zip(self.layer_sizes, self.layer_positions)):
            # Distribute nodes vertically, centered
            if layer_size == 1:
                y_positions = [4.0]  # Center single nodes
            else:
                y_positions = np.linspace(1, 7, layer_size)
            
            positions[f'layer_{layer_idx}'] = [(x_pos, y) for y in y_positions]
        
        return positions
    
    def setup_visualization(self) -> None:
        """Create the initial network visualization."""
        # Create nodes for each layer
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            layer_nodes = []
            positions = self.node_positions[f'layer_{layer_idx}']
            
            for node_idx in range(layer_size):
                x, y = positions[node_idx]
                
                # Node appearance based on layer type
                if layer_idx == 0:  # Input layer
                    color = 'lightblue'
                    size = 0.15
                    labels = ['Pos', 'Vel', 'Ang', 'AngVel']
                    label = labels[node_idx] if node_idx < len(labels) else f'In{node_idx}'
                elif layer_idx == len(self.layer_sizes) - 1:  # Output layer
                    color = 'lightcoral'
                    size = 0.2
                    labels = ['Left', 'Right']
                    label = labels[node_idx] if node_idx < len(labels) else f'Out{node_idx}'
                else:  # Hidden layers
                    color = 'lightgreen'
                    size = 0.08
                    label = f'H{node_idx}'
                
                # Create node circle
                circle = Circle((x, y), size, color=color, alpha=0.7, zorder=2)
                self.ax.add_patch(circle)
                layer_nodes.append(circle)
                
                # Add label
                if layer_idx in [0, len(self.layer_sizes) - 1] or layer_size <= 10:
                    self.ax.text(x, y-size-0.2, label, ha='center', va='top', 
                               fontsize=8, fontweight='bold')
            
            self.nodes.append(layer_nodes)
        
        # Create connections between layers
        for layer_idx in range(len(self.layer_sizes) - 1):
            current_positions = self.node_positions[f'layer_{layer_idx}']
            next_positions = self.node_positions[f'layer_{layer_idx+1}']
            
            layer_connections = []
            for i, (x1, y1) in enumerate(current_positions):
                node_connections = []
                for j, (x2, y2) in enumerate(next_positions):
                    # Create connection line
                    line = self.ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.1, 
                                      linewidth=0.5, zorder=1)[0]
                    node_connections.append(line)
                layer_connections.append(node_connections)
            
            self.connections.append(layer_connections)
        
        # Add weight statistics text
        self.weight_text = self.ax.text(0.5, 0.5, '', transform=self.ax.transAxes,
                                       bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor="white", alpha=0.8),
                                       verticalalignment='top', fontsize=10)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Input Layer'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=10, label='Hidden Layers'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=10, label='Output Layer')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    def update_visualization(self, episode: int, sample_state: Optional[np.ndarray] = None) -> None:
        """
        Update the visualization with current network state.
        
        Args:
            episode: Current training episode
            sample_state: Sample state for activation visualization
        """
        # Get current weights
        weights = self.network.get_layer_weights()
        
        # Update node colors based on activations
        if sample_state is not None:
            activations = self._get_sample_activations(sample_state)
            self._update_node_colors(activations)
        
        # Update connection weights
        self._update_connection_weights(weights)
        
        # Update statistics text
        stats_text = self._generate_stats_text(weights, episode)
        self.weight_text.set_text(stats_text)
        
        # Force redraw and flush events
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _get_sample_activations(self, state: np.ndarray) -> Dict[str, torch.Tensor]:
        """Get network activations for a sample state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.network.get_activations(state_tensor)
    
    def _update_node_colors(self, activations: Dict[str, torch.Tensor]) -> None:
        """Update node colors based on activation strength."""
        # Input layer - use actual state values
        input_values = activations.get('layer_0', torch.zeros(1, 4)).squeeze().numpy()
        for i, node in enumerate(self.nodes[0]):
            if i < len(input_values):
                # Normalize to [0, 1] for color intensity
                intensity = np.tanh(abs(input_values[i])) * 0.7 + 0.3
                color = plt.cm.Blues(intensity)
                node.set_facecolor(color)
        
        # Hidden layers - use activation strength
        for layer_idx in range(1, len(self.nodes) - 1):
            layer_key = f'layer_{layer_idx-1}'  # Activations are 0-indexed
            if layer_key in activations:
                layer_activations = activations[layer_key].squeeze().numpy()
                layer_nodes = self.nodes[layer_idx]
                
                # Only show a subset for large hidden layers
                num_nodes_to_show = min(len(layer_nodes), 20)
                selected_indices = np.linspace(0, len(layer_activations)-1, 
                                             num_nodes_to_show, dtype=int)
                
                for i, node_idx in enumerate(selected_indices):
                    if i < len(layer_nodes):
                        activation = layer_activations[node_idx]
                        intensity = np.tanh(activation) * 0.7 + 0.3
                        color = plt.cm.Greens(intensity)
                        layer_nodes[i].set_facecolor(color)
        
        # Output layer - Q-values
        q_values = activations.get('q_values', torch.zeros(1, 2)).squeeze().numpy()
        for i, node in enumerate(self.nodes[-1]):
            if i < len(q_values):
                # Color based on Q-value magnitude
                q_val = q_values[i]
                intensity = np.tanh(abs(q_val) / 10) * 0.7 + 0.3  # Scale Q-values
                color = plt.cm.Reds(intensity) if q_val > 0 else plt.cm.Blues(intensity)
                node.set_facecolor(color)
    
    def _update_connection_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Update connection thickness based on weight magnitude."""
        for layer_idx, layer_connections in enumerate(self.connections):
            weight_key = f'layer_{layer_idx}'
            if weight_key in weights:
                weight_matrix = weights[weight_key].numpy()
                
                # Normalize weights for visualization
                max_weight = np.max(np.abs(weight_matrix)) + 1e-6
                
                for i, node_connections in enumerate(layer_connections):
                    for j, connection in enumerate(node_connections):
                        if i < weight_matrix.shape[1] and j < weight_matrix.shape[0]:
                            weight = weight_matrix[j, i]  # Transposed for matrix indexing
                            
                            # Update line properties
                            thickness = abs(weight) / max_weight * 3 + 0.1
                            alpha = min(abs(weight) / max_weight * 0.5 + 0.05, 0.3)
                            color = 'red' if weight > 0 else 'blue'
                            
                            connection.set_linewidth(thickness)
                            connection.set_alpha(alpha)
                            connection.set_color(color)
    
    def _generate_stats_text(self, weights: Dict[str, torch.Tensor], episode: int) -> str:
        """Generate statistics text for the visualization."""
        stats = []
        stats.append(f"Episode: {episode}")
        stats.append(f"Network Layers: {' → '.join(map(str, self.layer_sizes))}")
        
        # Weight statistics
        all_weights = torch.cat([w.flatten() for w in weights.values()])
        stats.append(f"Weight Stats:")
        stats.append(f"  Mean: {all_weights.mean().item():.4f}")
        stats.append(f"  Std: {all_weights.std().item():.4f}")
        stats.append(f"  Range: [{all_weights.min().item():.3f}, {all_weights.max().item():.3f}]")
        
        return '\n'.join(stats)


class WeightHeatmapVisualizer:
    """
    Shows layer weights as evolving heatmaps.
    
    Each layer becomes a color grid where you can watch specific neurons
    specialize and weight patterns emerge over training.
    """
    
    def __init__(self, network: DQNetwork, figsize: Tuple[int, int] = (15, 10)):
        """Initialize the weight heatmap visualizer."""
        self.network = network
        self.weights_history = []
        
        # Get layer information
        self.layer_weights = network.get_layer_weights()
        self.num_layers = len(self.layer_weights)
        
        # Create subplots
        rows = 2
        cols = (self.num_layers + 1) // 2
        self.fig, self.axes = plt.subplots(rows, cols, figsize=figsize)
        if self.num_layers == 1:
            self.axes = [self.axes]
        elif rows == 1:
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flatten()
        
        self.fig.suptitle('DQN Weight Evolution Heatmaps\nWatch patterns emerge!', 
                         fontsize=16, fontweight='bold')
        
        # Initialize heatmaps
        self.heatmaps = []
        self.colorbars = []
        self.setup_heatmaps()
    
    def setup_heatmaps(self) -> None:
        """Create initial heatmap visualizations."""
        for i, (layer_name, weights) in enumerate(self.layer_weights.items()):
            if i >= len(self.axes):
                break
            
            ax = self.axes[i]
            weight_matrix = weights.cpu().numpy()
            
            # Create heatmap
            im = ax.imshow(weight_matrix, cmap='RdBu_r', aspect='auto', 
                          vmin=-1, vmax=1, interpolation='nearest')
            
            ax.set_title(f'Layer {i+1} Weights\n({weight_matrix.shape[0]}x{weight_matrix.shape[1]})')
            ax.set_xlabel('Input Neurons')
            ax.set_ylabel('Output Neurons')
            
            # Add colorbar
            cbar = self.fig.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label('Weight Value')
            
            self.heatmaps.append(im)
            self.colorbars.append(cbar)
        
        # Hide unused axes
        for i in range(len(self.layer_weights), len(self.axes)):
            self.axes[i].set_visible(False)
        
        plt.tight_layout()
    
    def update_heatmaps(self, episode: int) -> None:
        """Update heatmaps with current weights."""
        current_weights = self.network.get_layer_weights()
        self.weights_history.append(current_weights)
        
        # Update each heatmap
        for i, (layer_name, weights) in enumerate(current_weights.items()):
            if i >= len(self.heatmaps):
                break
            
            weight_matrix = weights.cpu().numpy()
            
            # Auto-scale color range based on current weights
            vmax = max(abs(weight_matrix.min()), abs(weight_matrix.max()), 0.1)
            
            self.heatmaps[i].set_data(weight_matrix)
            self.heatmaps[i].set_clim(-vmax, vmax)
            
            # Update title with statistics
            mean_weight = weight_matrix.mean()
            std_weight = weight_matrix.std()
            self.axes[i].set_title(f'Layer {i+1} Weights (Ep: {episode})\n'
                                  f'Mean: {mean_weight:.3f}, Std: {std_weight:.3f}')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class TrainingDashboard:
    """
    Complete real-time training dashboard combining all visualizations.
    
    This creates a multi-panel view showing:
    - Network architecture evolution
    - Weight heatmaps
    - Training metrics
    - Q-value progression
    """
    
    def __init__(self, agent: DQNAgent, update_frequency: int = 10):
        """
        Initialize the complete training dashboard.
        
        Args:
            agent: The DQN agent to visualize
            update_frequency: Update visualizations every N episodes
        """
        self.agent = agent
        self.update_frequency = update_frequency
        
        # Training data
        self.episodes = []
        self.rewards = []
        self.losses = []
        self.q_values = []
        self.epsilons = []
        
        # Create visualizers
        self.network_viz = NetworkArchitectureVisualizer(agent.q_network)
        self.heatmap_viz = WeightHeatmapVisualizer(agent.q_network)
        
        # Training metrics plot
        self.fig_metrics, self.axes_metrics = plt.subplots(2, 2, figsize=(12, 8))
        self.fig_metrics.suptitle('DQN Training Metrics Dashboard', fontsize=16, fontweight='bold')
        
        self.setup_metrics_plots()
        
        # Configure matplotlib for interactive mode
        plt.ion()  # Turn on interactive mode
        
        # Show all windows with explicit titles and force them to display
        print("🎨 Creating visualization windows...")
        
        # Create windows in sequence with pauses - Network Architecture FIRST
        print("   🧠 Creating network architecture window...")
        self.network_viz.fig.canvas.manager.set_window_title("1️⃣ DQN Network Architecture - NODES & CONNECTIONS")
        
        # Force the window to the front and make it prominent  
        plt.figure(self.network_viz.fig.number)
        self.network_viz.ax.set_facecolor('#f8f8f8')  # Very light gray background
        self.network_viz.fig.show()
        plt.draw()
        plt.pause(0.3)  # Pause to let window appear
        print("   ✅ Network architecture window created (look for nodes & lines!)")
        
        # Weight heatmaps window - SECOND
        print("   🔥 Creating weight heatmap window...")
        self.heatmap_viz.fig.canvas.manager.set_window_title("2️⃣ DQN Weight Heatmaps - EVOLUTION")
        plt.figure(self.heatmap_viz.fig.number)
        self.heatmap_viz.fig.show()
        plt.pause(0.3)
        print("   ✅ Weight heatmap window created")
        
        # Training metrics window - THIRD
        print("   📈 Creating training metrics window...")
        self.fig_metrics.canvas.manager.set_window_title("3️⃣ DQN Training Metrics - DASHBOARD")  
        plt.figure(self.fig_metrics.number)
        self.fig_metrics.show()
        plt.pause(0.3)
        print("   ✅ Training metrics window created")
        
        # Final pause and bring network architecture to front
        print("   🎯 Bringing network architecture to front...")
        plt.figure(self.network_viz.fig.number)
        plt.pause(0.5)
        
        print("🎨 Real-time visualization dashboard started!")
        print(f"   📊 Updates every {update_frequency} episodes")
        print("   🧠 Network architecture window (nodes & connections)")
        print("   🔥 Weight heatmap window (layer matrices)") 
        print("   📈 Training metrics window (rewards, loss, etc.)")
        print("   ⚠️  Keep all 3 windows visible during training!")
    
    def setup_metrics_plots(self) -> None:
        """Setup the training metrics plots."""
        # Episode rewards
        self.axes_metrics[0, 0].set_title('Episode Rewards')
        self.axes_metrics[0, 0].set_xlabel('Episode')
        self.axes_metrics[0, 0].set_ylabel('Total Reward')
        self.axes_metrics[0, 0].grid(True, alpha=0.3)
        
        # Training loss
        self.axes_metrics[0, 1].set_title('Training Loss')
        self.axes_metrics[0, 1].set_xlabel('Episode')
        self.axes_metrics[0, 1].set_ylabel('MSE Loss')
        self.axes_metrics[0, 1].set_yscale('log')
        self.axes_metrics[0, 1].grid(True, alpha=0.3)
        
        # Q-values
        self.axes_metrics[1, 0].set_title('Average Q-Values')
        self.axes_metrics[1, 0].set_xlabel('Episode')
        self.axes_metrics[1, 0].set_ylabel('Q-Value')
        self.axes_metrics[1, 0].grid(True, alpha=0.3)
        
        # Epsilon decay
        self.axes_metrics[1, 1].set_title('Exploration Rate (ε)')
        self.axes_metrics[1, 1].set_xlabel('Episode')
        self.axes_metrics[1, 1].set_ylabel('Epsilon')
        self.axes_metrics[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update_dashboard(self, episode: int, episode_reward: float, 
                        sample_state: Optional[np.ndarray] = None) -> None:
        """
        Update all visualizations.
        
        Args:
            episode: Current episode number
            episode_reward: Reward for this episode
            sample_state: Sample state for network visualization
        """
        # Update data
        self.episodes.append(episode)
        self.rewards.append(episode_reward)
        
        # Get agent statistics
        stats = self.agent.get_training_stats()
        if stats['avg_loss'] > 0:
            self.losses.append(stats['avg_loss'])
        if stats['avg_q_value'] > 0:
            self.q_values.append(stats['avg_q_value'])
        self.epsilons.append(stats['epsilon'])
        
        # Update visualizations at specified frequency
        if episode % self.update_frequency == 0 or episode <= 10:
            # Update network architecture
            self.network_viz.update_visualization(episode, sample_state)
            
            # Update weight heatmaps
            self.heatmap_viz.update_heatmaps(episode)
            
            # Update training metrics
            self._update_metrics_plots()
            
            print(f"📊 Dashboard updated at episode {episode}")
    
    def _update_metrics_plots(self) -> None:
        """Update the training metrics plots."""
        # Clear and redraw each plot
        for ax in self.axes_metrics.flat:
            ax.clear()
        
        self.setup_metrics_plots()
        
        # Plot episode rewards with moving average
        if len(self.rewards) > 0:
            self.axes_metrics[0, 0].plot(self.episodes, self.rewards, 'b-', alpha=0.7, label='Episode Reward')
            if len(self.rewards) >= 10:
                moving_avg = np.convolve(self.rewards, np.ones(10)/10, mode='valid')
                episodes_avg = self.episodes[9:]
                self.axes_metrics[0, 0].plot(episodes_avg, moving_avg, 'r-', linewidth=2, label='10-ep Average')
            self.axes_metrics[0, 0].legend()
            self.axes_metrics[0, 0].set_title(f'Episode Rewards (Latest: {self.rewards[-1]:.0f})')
        
        # Plot training loss
        if len(self.losses) > 0:
            loss_episodes = self.episodes[-len(self.losses):]
            self.axes_metrics[0, 1].plot(loss_episodes, self.losses, 'g-', alpha=0.7)
            self.axes_metrics[0, 1].set_title(f'Training Loss (Latest: {self.losses[-1]:.4f})')
        
        # Plot Q-values
        if len(self.q_values) > 0:
            q_episodes = self.episodes[-len(self.q_values):]
            self.axes_metrics[1, 0].plot(q_episodes, self.q_values, 'm-', alpha=0.7)
            self.axes_metrics[1, 0].set_title(f'Average Q-Values (Latest: {self.q_values[-1]:.2f})')
        
        # Plot epsilon decay
        if len(self.epsilons) > 0:
            self.axes_metrics[1, 1].plot(self.episodes, self.epsilons, 'orange', linewidth=2)
            self.axes_metrics[1, 1].set_title(f'Exploration Rate (ε = {self.epsilons[-1]:.3f})')
        
        plt.tight_layout()
        self.fig_metrics.canvas.draw()
        self.fig_metrics.canvas.flush_events()
    
    def save_dashboard(self, save_dir: str = "outputs/visualizations") -> None:
        """Save all visualizations to files."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Save network architecture
        self.network_viz.fig.savefig(save_path / "network_architecture.png", 
                                   dpi=300, bbox_inches='tight')
        
        # Save weight heatmaps
        self.heatmap_viz.fig.savefig(save_path / "weight_heatmaps.png", 
                                   dpi=300, bbox_inches='tight')
        
        # Save training metrics
        self.fig_metrics.savefig(save_path / "training_metrics.png", 
                               dpi=300, bbox_inches='tight')
        
        print(f"💾 Visualizations saved to {save_path}/")


def create_sample_state() -> np.ndarray:
    """Create a sample Cart-Pole state for visualization."""
    return np.array([0.1, -0.5, 0.02, 0.3])  # Slightly off-center, moving


if __name__ == "__main__":
    # Demo of visualization components
    print("🧪 Testing Neural Network Visualization Components...")
    
    # Create a sample network
    network = DQNetwork(state_size=4, action_size=2, hidden_sizes=(128, 128))
    
    # Test network architecture visualizer
    print("📊 Creating network architecture visualization...")
    arch_viz = NetworkArchitectureVisualizer(network)
    
    # Simulate some training updates
    sample_state = create_sample_state()
    for episode in range(0, 100, 20):
        print(f"   Updating visualization for episode {episode}")
        arch_viz.update_visualization(episode, sample_state)
        time.sleep(0.5)
    
    print("✅ Network visualization test complete!")
    print("   Close the plot window to continue...")
    plt.show()