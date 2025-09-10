#!/usr/bin/env python3
"""
Cart-Pole Environment Exploration

This script helps us understand the Cart-Pole environment:
- State space (4 values: position, velocity, angle, angular velocity)
- Action space (2 actions: left, right)
- Reward structure (+1 for each step)
- Termination conditions (angle > ±12°, position > ±2.4)
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def explore_environment():
    """Explore the Cart-Pole environment to understand its properties."""
    print("🔍 Exploring Cart-Pole Environment")
    print("=" * 50)

    # Create environment
    env = gym.make("CartPole-v1")

    # Print basic environment info
    print(f"Environment ID: {env.spec.id}")
    print(f"Max episode steps: {env.spec.max_episode_steps}")
    print()

    # Explore state space
    print("📊 State Space (Observation Space):")
    print(f"  Type: {type(env.observation_space)}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Data type: {env.observation_space.dtype}")
    print(f"  Low bounds: {env.observation_space.low}")
    print(f"  High bounds: {env.observation_space.high}")
    print()

    print("State components:")
    state_names = [
        "Cart Position",
        "Cart Velocity",
        "Pole Angle (radians)",
        "Pole Angular Velocity"
    ]
    for i, name in enumerate(state_names):
        low = env.observation_space.low[i]
        high = env.observation_space.high[i]
        print(f"  {i}: {name:<25} Range: [{low:>8.3f}, {high:>8.3f}]")
    print()

    # Explore action space
    print("🎮 Action Space:")
    print(f"  Type: {type(env.action_space)}")
    print(f"  Number of actions: {env.action_space.n}")
    print("Action meanings:")
    print("  0: Push cart to the LEFT")
    print("  1: Push cart to the RIGHT")
    print()

    # Termination conditions
    print("🏁 Termination Conditions:")
    print("  1. Pole angle > ±12° (±0.2095 radians)")
    print("  2. Cart position > ±2.4 units")
    print("  3. Episode length reaches 500 steps (CartPole-v1)")
    print()

    # Reward structure
    print("🏆 Reward Structure:")
    print("  +1 for every step taken")
    print("  Episode terminates when pole falls or cart moves too far")
    print("  Maximum possible reward: 500 (if episode runs full length)")
    print()

    # Run a few episodes to see actual behavior
    print("🎲 Running sample episodes to observe states...")

    total_rewards = []
    all_states = []

    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        episode_states = []

        for _ in range(env.spec.max_episode_steps):
            # Take random action
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            episode_states.append(state.copy())

            if terminated or truncated:
                break

            state = next_state

        total_rewards.append(episode_reward)
        all_states.extend(episode_states)
        print(f"  Episode {episode + 1}: {int(episode_reward)} steps")

    print()
    print("Random policy performance:")
    print(f"  Average episode length: {np.mean(total_rewards):.1f} steps")
    print(f"  Min/Max: {min(total_rewards)}/{max(total_rewards)} steps")
    print()

    # Analyze state distributions
    print("📈 Creating state distribution analysis...")
    all_states = np.array(all_states)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Cart-Pole State Distributions (Random Policy)', fontsize=14)

    for i, (ax, name) in enumerate(zip(axes.flat, state_names, strict=False)):
        ax.hist(all_states[:, i], bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/cartpole_state_distributions.png', dpi=300, bbox_inches='tight')
    print("📊 State distribution plot saved to 'outputs/cartpole_state_distributions.png'")

    # Show correlations between states and actions
    print()
    print("🔗 Key Insights for RL:")
    print("  - State space is continuous but bounded")
    print("  - Action space is discrete (2 actions)")
    print("  - Sparse reward (+1 per step until failure)")
    print("  - Episode can be 'solved' by balancing pole for 500 steps")
    print("  - Random policy achieves ~20-50 steps on average")
    print()

    env.close()
    print("✅ Environment exploration complete!")


if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs('outputs', exist_ok=True)

    explore_environment()
