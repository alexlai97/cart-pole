"""
Experience Replay Buffer for Deep Q-Learning

This module implements the replay buffer that makes DQN possible by solving
the sample efficiency problem we discovered in Q-learning.

Q-Learning Problem: Only explored 1.3% of state space (2,000 out of 160,000 states)
DQN Solution: Reuse transitions multiple times + break sequential correlations

Learning Goals:
- Understand why we need to store and replay experiences
- See how random sampling breaks correlation in sequential data  
- Experience the sample efficiency gains from reusing data
- Foundation for all modern deep RL algorithms
"""

import random
from collections import deque, namedtuple
from typing import Any, List, Optional, Tuple

import numpy as np


# Define experience tuple for type safety and clarity
Experience = namedtuple('Experience', [
    'state',     # Current state (4 values for Cart-Pole)
    'action',    # Action taken (0 or 1)
    'reward',    # Reward received (typically +1)
    'next_state',# Resulting state
    'done'       # Episode termination flag
])


class ReplayBuffer:
    """
    Circular buffer storing agent experiences for training deep RL agents.
    
    The replay buffer solves two critical problems from Q-learning:
    1. Sample Efficiency: Reuse precious data multiple times
    2. Correlation Breaking: Random sampling destroys sequential correlation
    
    Key Insights:
    - Q-learning: Each transition used once, then discarded
    - DQN: Store 10,000+ transitions, sample randomly for training
    - Result: 10x+ better sample efficiency
    """
    
    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducible sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)  # Circular buffer - old data auto-removed
        self.position = 0
        
        if seed is not None:
            random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a single experience to the buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode terminated
        """
        experience = Experience(
            state=state.copy(),      # Copy to avoid reference issues
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done
        )
        
        # Deque automatically handles capacity overflow
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                               np.ndarray, np.ndarray]:
        """
        Sample a random mini-batch of experiences for training.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
            
        Raises:
            ValueError: If not enough experiences stored for sampling
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} experiences. "
                           f"Buffer only contains {len(self.buffer)} experiences.")
        
        # Random sampling is KEY - breaks temporal correlation
        experiences = random.sample(self.buffer, batch_size)
        
        # Unpack experiences into separate arrays for neural network training
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences], dtype=bool)
        
        return states, actions, rewards, next_states, dones
    
    def can_sample(self, batch_size: int) -> bool:
        """
        Check if buffer contains enough experiences for sampling.
        
        Args:
            batch_size: Desired batch size
            
        Returns:
            True if sampling is possible
        """
        return len(self.buffer) >= batch_size
    
    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        return len(self.buffer)
    
    def get_buffer_stats(self) -> dict[str, Any]:
        """
        Get statistics about the replay buffer contents.
        
        Returns:
            Dictionary with buffer statistics
        """
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "avg_reward": 0.0,
                "action_distribution": np.array([0.0, 0.0])
            }
        
        # Extract data for analysis
        rewards = [exp.reward for exp in self.buffer]
        actions = [exp.action for exp in self.buffer]
        
        return {
            "size": len(self.buffer),
            "capacity": self.capacity, 
            "utilization": len(self.buffer) / self.capacity,
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "action_distribution": np.bincount(actions, minlength=2) / len(actions),
            "done_percentage": np.mean([exp.done for exp in self.buffer]) * 100
        }
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()
        self.position = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Advanced replay buffer with prioritized sampling based on TD error.
    
    Standard replay buffer samples uniformly. This version samples more
    frequently from experiences with high TD error (surprising transitions).
    
    Note: Implementing for educational completeness, but simple uniform
    sampling is sufficient for Cart-Pole and easier to understand.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, 
                 beta_start: float = 0.4, beta_frames: int = 100000,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta_start: Initial importance sampling weight
            beta_frames: Frames to anneal beta to 1.0
            seed: Random seed
        """
        super().__init__(capacity, seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Priority tree for efficient sampling (simplified version)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add experience with maximum priority (will be updated after training)."""
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Sample experiences based on priorities.
        
        Returns:
            Same as parent class plus weights and indices for priority updates
        """
        # For educational simplicity, fall back to uniform sampling
        # Real implementation would use sum tree for efficient priority sampling
        states, actions, rewards, next_states, dones = super().sample(batch_size)
        
        # Compute importance sampling weights
        beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame / self.beta_frames)
        weights = np.ones(batch_size, dtype=np.float32)  # Simplified
        indices = list(range(batch_size))  # Simplified
        
        self.frame += 1
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities for sampled experiences based on TD errors.
        
        Args:
            indices: Indices of sampled experiences
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)


def test_replay_buffer() -> None:
    """
    Test replay buffer functionality with sample Cart-Pole data.
    
    This demonstrates the key concepts and verifies correct operation.
    """
    print("🔬 Testing Replay Buffer Functionality...")
    print("=" * 50)
    
    # Create buffer
    buffer = ReplayBuffer(capacity=5, seed=42)  # Small for testing
    
    # Add some fake Cart-Pole experiences
    fake_experiences = [
        (np.array([0.1, 0.2, 0.01, -0.1]), 0, 1.0, np.array([0.15, 0.15, 0.005, -0.05]), False),
        (np.array([0.15, 0.15, 0.005, -0.05]), 1, 1.0, np.array([0.2, 0.1, 0.0, 0.0]), False),
        (np.array([0.2, 0.1, 0.0, 0.0]), 0, 1.0, np.array([0.18, 0.05, -0.005, 0.02]), False),
        (np.array([0.18, 0.05, -0.005, 0.02]), 1, 1.0, np.array([0.16, 0.0, -0.01, 0.05]), False),
        (np.array([0.16, 0.0, -0.01, 0.05]), 0, 0.0, np.array([0.0, 0.0, 0.0, 0.0]), True),  # Terminal
    ]
    
    print(f"📥 Adding {len(fake_experiences)} experiences to buffer...")
    for state, action, reward, next_state, done in fake_experiences:
        buffer.push(state, action, reward, next_state, done)
        print(f"   Buffer size: {len(buffer)}")
    
    # Test statistics
    stats = buffer.get_buffer_stats()
    print(f"\n📊 Buffer Statistics:")
    print(f"   Size: {stats['size']}/{stats['capacity']} ({stats['utilization']:.1%} full)")
    print(f"   Average reward: {stats['avg_reward']:.2f}")
    print(f"   Action distribution: Left {stats['action_distribution'][0]:.1%}, Right {stats['action_distribution'][1]:.1%}")
    print(f"   Terminal states: {stats['done_percentage']:.1f}%")
    
    # Test sampling
    if buffer.can_sample(3):
        print(f"\n🎲 Sampling 3 experiences...")
        states, actions, rewards, next_states, dones = buffer.sample(3)
        print(f"   States shape: {states.shape}")
        print(f"   Actions: {actions}")
        print(f"   Rewards: {rewards}")
        print(f"   Dones: {dones}")
        
        # Sample again to show randomness
        print(f"\n🎲 Sampling again (should be different due to randomness)...")
        states2, actions2, _, _, _ = buffer.sample(3)
        print(f"   Actions first sample:  {actions}")
        print(f"   Actions second sample: {actions2}")
        print(f"   Same order? {np.array_equal(actions, actions2)}")
    
    # Test capacity overflow
    print(f"\n📦 Testing capacity overflow (adding 2 more to 5-capacity buffer)...")
    buffer.push(np.array([0.5, 0.0, 0.0, 0.0]), 1, 1.0, np.array([0.4, 0.0, 0.0, 0.0]), False)
    buffer.push(np.array([0.4, 0.0, 0.0, 0.0]), 0, 1.0, np.array([0.3, 0.0, 0.0, 0.0]), False)
    print(f"   Buffer size after overflow: {len(buffer)} (should still be 5)")
    
    print(f"\n✅ Replay Buffer test completed successfully!")
    print(f"💡 Key Insight: Random sampling breaks temporal correlation")
    print(f"💡 Sample efficiency: Same data used multiple times for training")


if __name__ == "__main__":
    # Run tests if executed directly
    test_replay_buffer()