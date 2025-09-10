# Cart-Pole RL Project Context

## Project Purpose
This is a learning project to understand reinforcement learning through hands-on implementation. The user wants to learn AI/RL concepts by building agents for the Cart-Pole environment, with a focus on visualizing neural network evolution over time.

## Key Learning Goals
1. **Visual Learning**: The user specifically wants to SEE neural networks evolving - prioritize visualizations
2. **Incremental Progress**: Work through TODO.md items one at a time with the user
3. **Understanding Over Performance**: Focus on learning and understanding concepts, not just achieving high scores
4. **Interactive Experience**: User enjoys playing with the environment to build intuition

## Development Setup
- **Package Manager**: Use `uv` (Rust-based, fast Python package manager)
- **Linting/Formatting**: Ruff (all-in-one, very fast)
- **Type Checking**: Pyright or Mypy
- **Python Version**: Use latest stable (3.13+)
- **Environment**: Mac mini with fish shell

## Clean Project Structure
```
cart-pole/
├── agents/
│   ├── __init__.py
│   └── random_agent.py          # ✅ Baseline agent (23.3 steps avg)
├── utils/
│   ├── __init__.py
│   ├── interactive_play.py      # ✅ Consolidated play functionality
│   └── environment_explorer.py  # ✅ Environment analysis
├── visualization/
│   ├── __init__.py
│   └── plot_results.py          # ✅ Performance analysis & plotting
├── outputs/
│   ├── plots/                   # Generated visualizations
│   └── results/                 # JSON performance data
├── main.py                      # ✅ Clean entry point (no subprocess)
├── pyproject.toml              # Project dependencies & config
├── ruff.toml                   # Linting configuration
├── PLAN.md                     # Overall learning roadmap
├── TODO.md                     # Detailed task list
└── CLAUDE.md                   # This file - project context
```

## Coding Guidelines
1. **Simplicity First**: Start with the simplest implementation that works
2. **Clean Architecture**: No subprocess calls, proper imports, DRY principle
3. **Heavy Comments**: Since this is for learning, explain WHY not just WHAT
4. **Visualization Everything**: Every algorithm should have associated visualizations
5. **Type Hints**: Use type hints for better understanding
6. **Incremental Complexity**: Build from simple to complex

## Current Status
✅ **Environment Setup Complete** - uv, dependencies, linting configured
✅ **Random Agent Baseline** - 23.3 ± 11.5 steps average performance
✅ **Visualization System** - Performance plotting and analysis
✅ **Interactive Play** - Real-time and turn-based gameplay
✅ **Clean Codebase** - Reorganized, no redundant scripts

## Available Commands
```bash
# Main entry point with all functionality
python main.py --agent random --episodes 100    # Run random agent
python main.py --visualize                      # Analyze saved results
python main.py --explore                        # Explore environment
python main.py --play realtime                  # Real-time A/D gameplay
python main.py --play simple                    # Turn-based gameplay
python main.py --quick                          # Interactive menu

# Development commands
ruff check --fix .                              # Lint and fix code
pyright                                         # Type checking
```

## Algorithm Implementation Order
Following the TODO.md structure:
1. ✅ **Random Agent** - Baseline established (23.3 steps)
2. **Rule-based Agent** - Simple heuristics (beat random)
3. **Q-Learning** - Tabular reinforcement learning
4. **DQN** - Deep learning begins
5. **REINFORCE** - Policy gradient methods
6. **A2C** - Actor-critic architecture
7. **PPO** - State-of-the-art policy optimization

## Visualization Priorities
1. ✅ **Training curves** - Episode rewards over time
2. ✅ **Performance analysis** - Statistics and distributions
3. **Q-value heatmaps** - For Q-learning visualization
4. **Network weights evolution** - Deep learning visualization
5. **Policy distributions** - How action probabilities change
6. **Real-time dashboard** - Interactive training monitor

## Interactive Testing Goals
- **Agent Teasing Feature** - Apply disturbances to trained agents
- **Real-time Performance** - Watch agents handle unexpected situations
- **Algorithm Comparison** - Side-by-side robustness testing
- **Recovery Analysis** - Measure adaptation to perturbations

## Important Reminders
- The user is on a Mac mini with fish shell
- The user wants to LEARN - explain concepts as we implement
- Each task in TODO.md has a **Learning Goal** - make sure to address it
- Visualizations are KEY - the user wants to see networks evolving
- Work on one TODO item at a time unless asked otherwise
- Interactive functionality needs real terminal (input issues in headless mode)

## Baseline Performance
- **Random Agent**: 23.3 ± 11.5 steps (0% success rate)
- **Success Threshold**: 195+ steps average to "solve" Cart-Pole
- **Challenge**: All learning algorithms must significantly beat 23.3 steps

## Questions to Keep in Mind
- What makes this algorithm different from the previous one?
- What are the failure modes we might encounter?
- How can we visualize what the agent is learning?
- What hyperparameters matter most?
- How robust is the agent to disturbances?

## Notes from Research
- **uv** is 10-100x faster than pip and handles virtual environments
- **Ruff** replaces Black, isort, Flake8, and more - use it for everything
- **Pyright** is faster than Mypy for type checking
- Modern Python (2025) emphasizes speed without sacrificing features
- Cart-Pole physics: Position, velocity, angle, angular velocity → left/right actions