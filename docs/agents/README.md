# Cart-Pole Agents: A Beginner's Guide

This directory contains educational documentation about different types of agents that can play the Cart-Pole game. If you're new to reinforcement learning (RL), this is your starting point!

## What is Cart-Pole?

Cart-Pole is a classic control problem where you need to balance a pole on top of a moving cart. Think of balancing a broomstick on your palm while walking around.

### The Environment

**Goal**: Keep the pole upright as long as possible
**Actions**: Push the cart left (0) or right (1) 
**Observation**: You see 4 numbers that describe the current state:
- Cart position (how far left/right the cart is)
- Cart velocity (how fast the cart is moving)
- Pole angle (how much the pole is tilted)
- Pole angular velocity (how fast the pole is rotating)

### The Reward System

Cart-Pole uses a **sparse reward** system:
- **+1 reward** for every timestep the pole stays upright
- **Episode ends** when:
  - Pole tilts more than 12° from vertical
  - Cart moves more than 2.4 units from center
  - 500 timesteps pass (you "solve" Cart-Pole!)

### Why This is Hard

Unlike games with clear "good" and "bad" moves, Cart-Pole only tells you when you've failed. The agent must learn from:
- Delayed consequences (bad moves don't fail immediately)
- Sparse feedback (only +1 or game over)
- Continuous state space (infinite possible positions/velocities)
- Physics dynamics (actions have momentum effects)

## Agent Types in This Project

### 1. Random Agent 🎲
**File**: `agents/random_agent.py`
**Strategy**: Completely random actions
**Performance**: ~23 steps average
**Purpose**: Baseline to beat

### 2. Rule-Based Agent 🧠
**File**: `agents/rule_based_agent.py`  
**Strategy**: Simple heuristic rules
**Performance**: ~44 steps average
**Purpose**: Show value of domain knowledge

### 3. Q-Learning Agent 🎯
**File**: `agents/q_learning_agent.py`
**Strategy**: Learn state-action values with Q-tables
**Performance**: ~28 steps average
**Purpose**: First true RL algorithm - shows limitations

### 4. Deep Q-Network (DQN) 🧠🔗
**File**: `agents/dqn_agent.py`
**Strategy**: Neural network Q-values with experience replay
**Performance**: **~286 steps average (SOLVES Cart-Pole!)**
**Purpose**: Revolutionary deep RL breakthrough

## Learning Path

1. **Start Here**: Understand [Random](random-agent.md) and [Rule-Based](rule-based-agent.md) agents
2. **Foundation**: Learn [Q-Learning](q-learning-agent.md) fundamentals and limitations  
3. **Revolution**: Experience the [DQN](dqn-agent.md) breakthrough with neural networks
4. **Advanced**: Policy gradient methods (REINFORCE, A2C, PPO) *(Coming Soon)*

Each agent builds on concepts from the previous ones, so don't skip ahead!

## Key Concepts

**Agent**: The "brain" that decides what action to take
**State**: Current situation (4 numbers in Cart-Pole)
**Action**: What the agent chooses to do (left/right)
**Reward**: Feedback from environment (+1 per timestep)
**Episode**: One complete game (start to pole falling)
**Policy**: The agent's strategy for choosing actions

---

*Continue reading the specific agent guides to dive deeper into each approach!*