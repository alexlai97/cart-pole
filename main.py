#!/usr/bin/env python3
"""
Cart-Pole Reinforcement Learning Project

Main entry point for training and evaluating RL agents on the Cart-Pole environment.
This script serves as the central hub for our learning journey through various RL algorithms.

Usage:
    python main.py --agent random --episodes 100
    python main.py --agent random --render
    python main.py --visualize
"""

import argparse


def run_random_agent(episodes: int, render: bool) -> None:
    """Run the random agent."""
    print("🎲 Running Random Agent...")
    from agents.random_agent import evaluate_random_agent, print_results

    results = evaluate_random_agent(num_episodes=episodes, render=render)
    print_results(results)


def run_rule_based_agent(episodes: int, render: bool) -> None:
    """Run the rule-based agent."""
    print("🧠 Running Rule-Based Agent...")
    from agents.rule_based_agent import evaluate_rule_based_agent, print_results, compare_with_random

    results = evaluate_rule_based_agent(num_episodes=episodes, render=render)
    print_results(results)
    compare_with_random(results)


def run_q_learning_agent(episodes: int, render: bool) -> None:
    """Run the Q-learning agent."""
    print("🧮 Running Q-Learning Agent...")
    from agents.q_learning_agent import train_q_learning_agent

    results = train_q_learning_agent(episodes=episodes, render=render)
    
    print(f"\n🎯 Q-Learning vs Baselines:")
    print(f"   Random Agent:     23.3 ± 11.5 steps")
    print(f"   Rule-Based Agent: 43.8 ±  8.7 steps")
    print(f"   Q-Learning Agent: {results['mean_reward']:.1f} ± {results['std_reward']:.1f} steps")
    
    improvement_over_random = ((results['mean_reward'] - 23.3) / 23.3) * 100
    improvement_over_rule = ((results['mean_reward'] - 43.8) / 43.8) * 100
    
    print(f"\n📈 Improvements:")
    print(f"   vs Random:     {improvement_over_random:+.1f}%")
    print(f"   vs Rule-Based: {improvement_over_rule:+.1f}%")


def run_visualization(args: str = None) -> None:
    """Run visualization analysis with flexible options.
    
    Args:
        args: Visualization arguments - can be:
            - None or 'all': Analyze all available agents
            - 'agent_name': Analyze specific agent
            - 'agent1,agent2': Compare multiple agents
    """
    print("📊 Running visualization analysis...")
    
    from visualization.plot_results import (
        analyze_agent, 
        analyze_all_agents, 
        compare_agents,
        discover_available_agents
    )
    
    if args is None or args == "all":
        # Analyze all available agents
        analyze_all_agents()
    elif "," in args:
        # Compare multiple agents
        agent_names = [name.strip() for name in args.split(",")]
        compare_agents(agent_names)
    else:
        # Analyze specific agent
        if not analyze_agent(args):
            print(f"\n💡 Available agents: {', '.join(discover_available_agents())}")
            print("💡 Run agents first with: python main.py --agent <agent_name> --episodes 100")


def run_interactive_play(play_mode: str) -> None:
    """Run interactive play mode."""
    if play_mode == "simple":
        print("🎮 Launching Simple Interactive Play (turn-based)...")
        print("Use this for thoughtful, step-by-step control")
        from utils.interactive_play import play_turn_based
        play_turn_based()
    elif play_mode == "realtime":
        print("🎮 Launching Real-time Interactive Play...")
        print("Use A/D keys for continuous control - focus the game window!")
        from utils.interactive_play import play_realtime
        play_realtime()


def run_state_analysis(agent_name: str = "random", episodes: int = 100) -> None:
    """Run state space analysis."""
    print(f"🔬 Running state analysis with {agent_name} agent...")
    
    if agent_name == "random":
        from agents.random_agent import RandomAgent
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        agent = RandomAgent(env.action_space)
        env.close()
    elif agent_name == "rule_based":
        from agents.rule_based_agent import RuleBasedAgent
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        agent = RuleBasedAgent(env.action_space)
        env.close()
    else:
        print(f"❌ Unknown agent: {agent_name}")
        return
    
    from utils.state_analyzer import analyze_environment
    analyze_environment(agent, episodes)


def run_quick_menu() -> None:
    """Run interactive quick menu for common actions."""
    while True:
        print("\n" + "🚀 Cart-Pole Quick Menu" + "\n" + "=" * 25)
        print("1. 🎮 Play real-time (A/D keys)")
        print("2. 🎲 Run random agent")
        print("3. 🧠 Run rule-based agent") 
        print("4. 🧮 Run Q-learning agent")
        print("5. 📊 View results")
        print("6. 🔍 Explore environment")
        print("7. 🔬 Analyze state space")
        print("8. 🎯 Play turn-based")
        print("9. ❌ Exit")

        choice = input("\nChoose option (1-9): ").strip()

        if choice == '1':
            run_interactive_play("realtime")
        elif choice == '2':
            run_random_agent(100, False)
        elif choice == '3':
            run_rule_based_agent(100, False)
        elif choice == '4':
            run_q_learning_agent(500, False)  # Fewer episodes for interactive use
        elif choice == '5':
            run_visualization()
        elif choice == '6':
            import os

            from utils.environment_explorer import explore_environment
            os.makedirs('outputs', exist_ok=True)
            explore_environment()
        elif choice == '7':
            run_state_analysis("random", 100)
        elif choice == '8':
            run_interactive_play("simple")
        elif choice == '9':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice, please try again")


def main():
    """Main entry point for the Cart-Pole RL project."""
    parser = argparse.ArgumentParser(
        description="Cart-Pole Reinforcement Learning Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --agent random --episodes 100    # Run random agent
  python main.py --agent q-learning --episodes 1000  # Train Q-learning
  python main.py --agent random --render          # Run with visualization  
  python main.py --visualize                      # Analyze all agents
  python main.py --visualize random              # Analyze specific agent
  python main.py --visualize random,rule_based   # Compare agents
  python main.py --explore                        # Explore environment
  python main.py --analyze random --episodes 200  # Analyze state space
  python main.py --play simple                    # Play turn-based (think each move)
  python main.py --play realtime                  # Play real-time (A/D keys)
  python main.py --quick                          # Quick interactive menu
        """
    )

    parser.add_argument(
        "--agent",
        choices=["random", "rule-based", "q-learning"],
        help="Which agent to run",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during episodes",
    )
    parser.add_argument(
        "--visualize",
        nargs="?",
        const="all",
        help="Run visualization analysis (no arg=all agents, agent_name=specific, 'agent1,agent2'=compare)",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Explore the Cart-Pole environment",
    )
    parser.add_argument(
        "--analyze",
        choices=["random", "rule_based"],
        help="Analyze state space with specified agent",
    )
    parser.add_argument(
        "--play",
        choices=["simple", "realtime"],
        help="Play Cart-Pole interactively (simple=turn-based, realtime=continuous)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick menu for common actions",
    )

    args = parser.parse_args()

    print("🚀 Welcome to Cart-Pole RL Learning Project!")
    print("=" * 50)

    if args.quick:
        run_quick_menu()
    elif args.visualize is not None:
        run_visualization(args.visualize)
    elif args.explore:
        print("🔍 Launching environment exploration...")
        import os

        from utils.environment_explorer import explore_environment
        os.makedirs('outputs', exist_ok=True)
        explore_environment()
    elif args.analyze:
        run_state_analysis(args.analyze, args.episodes)
    elif args.play:
        run_interactive_play(args.play)
    elif args.agent == "random":
        run_random_agent(args.episodes, args.render)
    elif args.agent == "rule-based":
        run_rule_based_agent(args.episodes, args.render)
    elif args.agent == "q-learning":
        run_q_learning_agent(args.episodes, args.render)
    else:
        print("Please specify what to do:")
        print("  --agent random       # Run random agent")
        print("  --agent rule-based   # Run rule-based agent")
        print("  --agent q-learning   # Run Q-learning agent")
        print("  --visualize          # Analyze results")
        print("  --explore            # Explore environment")
        print("  --analyze random     # Analyze state space")
        print("  --play simple        # Play turn-based")
        print("  --play realtime      # Play real-time")
        print("  --quick              # Interactive menu")
        print("  --help               # Show all options")


if __name__ == "__main__":
    main()
