# Cart-Pole RL Learning Tasks

## How to Use This File
- Mark tasks as `[ ]` (todo), `[x]` (done), or `[~]` (in progress)
- Add notes under each task as we learn
- Feel free to add new tasks or modify existing ones
- Each task should be self-contained and learnable in 1-2 hours

## 🚀 Project Setup
- [x] Set up Python environment with uv
  - Install uv package manager
  - Create virtual environment
  - Set up pyproject.toml
- [x] Install core dependencies
  - gymnasium[classic-control]
  - numpy
  - matplotlib for basic plots
- [x] Configure development tools
  - Set up Ruff for linting/formatting
  - Configure Pyright or Mypy for type checking
  - Create pre-commit hooks
- [x] Create basic project structure
  - Create directories: agents/, visualization/, utils/
  - Set up __init__.py files
  - Create main.py entry point

## 🎮 Phase 1: Environment Basics
- [x] Explore Cart-Pole environment
  - Print state space (4 values: position, velocity, angle, angular velocity)
  - Print action space (2 actions: left, right)
  - Understand reward structure (+1 for each step)
  - Learn termination conditions (angle > ±12°, position > ±2.4)
- [x] Create random agent
  - Implement random action selection
  - Run episodes and collect rewards
  - Calculate average performance over 100 episodes
  - **Learning Goal**: Establish baseline performance (~23 steps)
- [ ] Build simple rule-based agent
  - If pole tilting right, move right (and vice versa)
  - Compare with random baseline
  - **Learning Goal**: Can simple rules beat random?
- [ ] Create first visualization
  - Plot episode rewards over time
  - Show moving average
  - Save plots to outputs/
  - **Learning Goal**: How to measure improvement

## 📊 Phase 2: Data Collection & Analysis
- [ ] Build experience collector
  - Store (state, action, reward, next_state, done) tuples
  - Create replay buffer class
  - **Learning Goal**: Why do we need experience replay?
- [ ] Analyze state distributions
  - Plot histograms of each state component
  - Find correlations between states and actions
  - **Learning Goal**: Understanding the state space
- [ ] Create episode renderer
  - Save episode as video/gif
  - Visualize cart and pole positions
  - **Learning Goal**: Visual debugging

## 🧮 Phase 3: Q-Learning Implementation
- [ ] Implement state discretization
  - Convert continuous states to discrete bins
  - Experiment with different bin sizes
  - **Learning Goal**: Trade-off between granularity and table size
- [ ] Build Q-table
  - Initialize Q-table with zeros
  - Implement Q-value updates
  - **Learning Goal**: How Q-values represent expected returns
- [ ] Implement Q-learning algorithm
  - Bellman equation updates
  - Learning rate scheduling
  - **Learning Goal**: Temporal difference learning
- [ ] Add epsilon-greedy exploration
  - Start with high exploration (ε=1.0)
  - Decay over time
  - **Learning Goal**: Exploration vs exploitation
- [ ] Visualize Q-table evolution
  - Heatmap of Q-values
  - Show how values change over episodes
  - **Learning Goal**: How Q-values converge

## 🧠 Phase 4: Deep Q-Network (DQN)
- [ ] Design neural network architecture
  - Input: 4 state values
  - Hidden layers: experiment with sizes
  - Output: 2 Q-values (one per action)
  - **Learning Goal**: Network as function approximator
- [ ] Implement basic DQN
  - Forward pass for Q-value prediction
  - Loss function (MSE between predicted and target Q)
  - **Learning Goal**: Replacing tables with networks
- [ ] Add experience replay buffer
  - Store last N experiences
  - Sample random mini-batches
  - **Learning Goal**: Breaking correlation in data
- [ ] Implement target network
  - Separate network for stable targets
  - Periodic weight copying
  - **Learning Goal**: Preventing moving targets problem
- [ ] Visualize network learning
  - Plot network weights as heatmap
  - Show weight changes over time
  - Track loss convergence
  - **Learning Goal**: How networks learn features

## 🎯 Phase 5: Policy Gradient (REINFORCE)
- [ ] Understand policy networks
  - Output: action probabilities
  - Sampling actions from distribution
  - **Learning Goal**: Direct policy optimization
- [ ] Implement REINFORCE
  - Collect full episode before update
  - Calculate discounted returns
  - **Learning Goal**: Monte Carlo policy gradient
- [ ] Add baseline for variance reduction
  - Use average reward as baseline
  - Compare learning curves with/without
  - **Learning Goal**: Reducing gradient variance
- [ ] Visualize policy evolution
  - Plot action probabilities for different states
  - Show how policy becomes deterministic
  - **Learning Goal**: Policy convergence patterns

## 🎭 Phase 6: Actor-Critic (A2C)
- [ ] Design actor network
  - Outputs action probabilities
  - **Learning Goal**: Policy function
- [ ] Design critic network
  - Outputs state value V(s)
  - **Learning Goal**: Value function
- [ ] Implement advantage estimation
  - A(s,a) = R - V(s)
  - **Learning Goal**: Better gradient estimates
- [ ] Combine actor and critic training
  - Actor loss with advantages
  - Critic loss with TD error
  - **Learning Goal**: Simultaneous optimization
- [ ] Compare with REINFORCE
  - Learning speed
  - Final performance
  - Stability
  - **Learning Goal**: Benefits of actor-critic

## 🚀 Phase 7: PPO Implementation
- [ ] Understand PPO motivation
  - Problems with large policy updates
  - **Learning Goal**: Trust region methods
- [ ] Implement clipped objective
  - Ratio of new/old policies
  - Clipping to prevent large changes
  - **Learning Goal**: Stable policy updates
- [ ] Add value function clipping
  - Similar to policy clipping
  - **Learning Goal**: Stable value updates
- [ ] Implement PPO training loop
  - Multiple epochs per batch
  - Early stopping if KL divergence too large
  - **Learning Goal**: Sample efficiency
- [ ] Create comprehensive comparison
  - All algorithms on same plot
  - Training time comparison
  - Sample efficiency metrics
  - **Learning Goal**: Algorithm trade-offs

## 📈 Phase 8: Advanced Visualizations
- [ ] Build real-time training dashboard
  - Live reward plots
  - Current policy visualization
  - Network statistics
  - **Learning Goal**: Debugging RL in real-time
- [ ] Create network architecture visualizer
  - Show layer connections
  - Activation patterns
  - Gradient flow
  - **Learning Goal**: Understanding deep networks
- [ ] Implement attention/saliency maps
  - Which state features matter most?
  - How does this change over training?
  - **Learning Goal**: Interpretability
- [ ] Build interactive demo
  - Adjust hyperparameters live
  - Compare algorithms side-by-side
  - **Learning Goal**: Hyperparameter sensitivity

## 🔬 Experiments & Analysis
- [ ] Hyperparameter sensitivity study
  - Learning rate effects
  - Network size impact
  - Batch size trade-offs
  - **Learning Goal**: Tuning strategies
- [ ] Ablation studies
  - Remove experience replay
  - Remove target network
  - Remove baseline
  - **Learning Goal**: Component importance
- [ ] Generalization tests
  - Train on standard Cart-Pole
  - Test on modified versions
  - **Learning Goal**: Overfitting in RL
- [ ] Create final report
  - Algorithm comparison table
  - Best practices learned
  - Common pitfalls
  - **Learning Goal**: Consolidate knowledge

## 🎮 Interactive Agent Testing (Fun!)
- [ ] Build interactive agent tester
  - Load trained agents from different algorithms
  - Apply manual disturbances (mouse clicks to "push" the pole)
  - Real-time visualization of agent's internal state
  - **Learning Goal**: See how robust different agents are
- [ ] Create disturbance scenarios
  - Sudden impulse forces
  - Continuous "wind" forces
  - Random perturbations
  - **Learning Goal**: Test agent adaptability
- [ ] Agent comparison dashboard
  - Side-by-side comparison of multiple agents
  - Same disturbances applied to all
  - Performance metrics during stress testing
  - **Learning Goal**: Compare algorithm robustness
- [ ] Recovery analysis tool
  - Measure time to recover from disturbances
  - Track success rate under different stress levels
  - Visualize failure modes
  - **Learning Goal**: Quantify real-world performance

## 💡 Bonus Challenges
- [ ] Implement prioritized experience replay
- [ ] Add curiosity-driven exploration
- [ ] Try double DQN
- [ ] Implement dueling networks
- [ ] Create custom reward shaping
- [ ] Build ensemble of agents
- [ ] Add parallel environment training

## 📚 Learning Resources Tasks
- [ ] Read Sutton & Barto Chapter 1-3 (RL basics)
- [ ] Complete OpenAI Spinning Up tutorials
- [ ] Watch David Silver's RL lectures (1-4)
- [ ] Study key papers (DQN, A3C, PPO)

## Notes Section
<!-- Add your learning notes, observations, and questions here as we progress -->

### What I Learned Today:
- 

### Questions to Explore:
- 

### Ideas for Improvements:
-