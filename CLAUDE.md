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
├── .claude/
│   └── agents/                  # ✅ Custom Claude Code subagents
│       ├── rl-teacher.md        # ✅ RL concept explanations (opus, yellow)
│       ├── agent-documenter.md  # ✅ Documentation creation (sonnet, blue)
│       ├── viz-explainer.md     # ✅ Visualization explanations (sonnet, green)
│       └── code-analyzer.md     # ✅ Code analysis (sonnet, purple)
├── agents/
│   ├── __init__.py
│   ├── random_agent.py          # ✅ Baseline agent (23.3 steps avg)
│   └── rule_based_agent.py      # ✅ Heuristic approach
├── docs/
│   └── agents/                  # ✅ Comprehensive beginner documentation
│       ├── README.md            # ✅ Agent overview & learning path
│       ├── random-agent.md      # ✅ Random baseline deep dive
│       ├── rule-based-agent.md  # ✅ Heuristic strategy explanation
│       └── rewards-and-costs.md # ✅ Cart-Pole reward system guide
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
✅ **Rule-Based Agent** - Heuristic approach implemented (43.8 ± 8.7 steps, 88% improvement!)
✅ **Flexible Visualization System** - Agent-agnostic plotting with comparison capabilities
✅ **State Space Analyzer** - Deep dive into environment dynamics for Q-learning prep
✅ **Interactive Play** - Real-time and turn-based gameplay
✅ **Clean Codebase** - Reorganized, no redundant scripts
✅ **Comprehensive Documentation** - Beginner-friendly guides for all agents
✅ **Claude Code Subagents** - Specialized AI assistants for education

## Available Commands
```bash
# CRITICAL: Always use the project's virtual environment
source .venv/bin/activate                       # MUST activate first!

# Main entry point with all functionality
python main.py --agent random --episodes 100    # Run random agent
python main.py --visualize                      # Analyze saved results
python main.py --explore                        # Explore environment
python main.py --analyze random --episodes 200  # Analyze state space (NEW!)
python main.py --play realtime                  # Real-time A/D gameplay
python main.py --play simple                    # Turn-based gameplay
python main.py --quick                          # Interactive menu

# Development commands (with venv activated)
ruff check --fix .                              # Lint and fix code
pyright                                         # Type checking

# Alternative: Use uv run (auto-activates)
uv run python main.py --analyze random          # Auto-activates venv
```

## Algorithm Implementation Order
Following the TODO.md structure:
1. ✅ **Random Agent** - Baseline established (23.3 steps)
2. ✅ **Rule-based Agent** - Simple heuristics (implemented, documentation complete)
3. **Q-Learning** - Tabular reinforcement learning
4. **DQN** - Deep learning begins
5. **REINFORCE** - Policy gradient methods
6. **A2C** - Actor-critic architecture
7. **PPO** - State-of-the-art policy optimization

## Visualization Priorities
1. ✅ **Training curves** - Episode rewards over time with moving averages
2. ✅ **Performance analysis** - Statistics, distributions, success rates
3. ✅ **Agent comparison** - Side-by-side performance visualization
4. ✅ **Flexible CLI** - `--visualize`, `--visualize agent_name`, `--visualize agent1,agent2`
5. **Q-value heatmaps** - For Q-learning visualization
6. **Network weights evolution** - Deep learning visualization
7. **Policy distributions** - How action probabilities change
8. **Real-time dashboard** - Interactive training monitor

## Interactive Testing Goals
- **Agent Teasing Feature** - Apply disturbances to trained agents
- **Real-time Performance** - Watch agents handle unexpected situations
- **Algorithm Comparison** - Side-by-side robustness testing
- **Recovery Analysis** - Measure adaptation to perturbations

## Educational Documentation System
✅ **Comprehensive Agent Guides** - Each algorithm has detailed beginner documentation
✅ **Learning Path Structure** - Clear progression from simple to complex
✅ **Visual Learning Focus** - Descriptions of what learners will "see"
✅ **Real-World Analogies** - Complex concepts explained through familiar examples
✅ **Performance Baselines** - All agents compared to random (23.3 steps)
✅ **Code Deep Dives** - Annotated walkthroughs of implementations

## Claude Code Subagents
✅ **rl-teacher** (opus, yellow) - Explains RL concepts in beginner-friendly ways
✅ **agent-documenter** (sonnet, blue) - Creates consistent documentation for new agents
✅ **viz-explainer** (sonnet, green) - Interprets training curves and visualizations
✅ **code-analyzer** (sonnet, purple) - Analyzes implementations for key concepts

### Using Subagents
```bash
# Task: "Explain Q-learning to a beginner" → rl-teacher
# Task: "Document the new DQN agent" → agent-documenter  
# Task: "Explain this training curve" → viz-explainer
# Task: "Analyze the policy gradient code" → code-analyzer
```

## Important Reminders
- The user is on a Mac mini with fish shell
- **CRITICAL**: ALWAYS use the project's virtual environment (`source .venv/bin/activate` or `uv run`)
- **NEVER** work in wrong venv - check for warnings about mismatched VIRTUAL_ENV paths
- The user wants to LEARN - explain concepts as we implement
- Each task in TODO.md has a **Learning Goal** - make sure to address it
- Visualizations are KEY - the user wants to see networks evolving
- Work on one TODO item at a time unless asked otherwise
- Interactive functionality needs real terminal (input issues in headless mode)
- **Use subagents** for educational tasks - they maintain consistent beginner-friendly tone
- **Environment Check**: If you see venv warnings, STOP and use correct activation method

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