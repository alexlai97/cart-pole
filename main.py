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


def run_visualization() -> None:
    """Run visualization analysis."""
    print("📊 Running visualization analysis...")
    from visualization.plot_results import analyze_random_agent

    analyze_random_agent()


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


def run_quick_menu() -> None:
    """Run interactive quick menu for common actions."""
    while True:
        print("\n" + "🚀 Cart-Pole Quick Menu" + "\n" + "=" * 25)
        print("1. 🎮 Play real-time (A/D keys)")
        print("2. 🎲 Run random agent")
        print("3. 🧠 Run rule-based agent")
        print("4. 📊 View results")
        print("5. 🔍 Explore environment")
        print("6. 🎯 Play turn-based")
        print("7. ❌ Exit")

        choice = input("\nChoose option (1-7): ").strip()

        if choice == '1':
            run_interactive_play("realtime")
        elif choice == '2':
            run_random_agent(100, False)
        elif choice == '3':
            run_rule_based_agent(100, False)
        elif choice == '4':
            run_visualization()
        elif choice == '5':
            import os

            from utils.environment_explorer import explore_environment
            os.makedirs('outputs', exist_ok=True)
            explore_environment()
        elif choice == '6':
            run_interactive_play("simple")
        elif choice == '7':
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
  python main.py --agent random --render          # Run with visualization  
  python main.py --visualize                      # Analyze saved results
  python main.py --explore                        # Explore environment
  python main.py --play simple                    # Play turn-based (think each move)
  python main.py --play realtime                  # Play real-time (A/D keys)
  python main.py --quick                          # Quick interactive menu
        """
    )

    parser.add_argument(
        "--agent",
        choices=["random", "rule-based"],
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
        action="store_true",
        help="Run visualization analysis of saved results",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Explore the Cart-Pole environment",
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
    elif args.visualize:
        run_visualization()
    elif args.explore:
        print("🔍 Launching environment exploration...")
        import os

        from utils.environment_explorer import explore_environment
        os.makedirs('outputs', exist_ok=True)
        explore_environment()
    elif args.play:
        run_interactive_play(args.play)
    elif args.agent == "random":
        run_random_agent(args.episodes, args.render)
    elif args.agent == "rule-based":
        run_rule_based_agent(args.episodes, args.render)
    else:
        print("Please specify what to do:")
        print("  --agent random       # Run random agent")
        print("  --agent rule-based   # Run rule-based agent")
        print("  --visualize          # Analyze results")
        print("  --explore            # Explore environment")
        print("  --play simple        # Play turn-based")
        print("  --play realtime      # Play real-time")
        print("  --quick              # Interactive menu")
        print("  --help               # Show all options")


if __name__ == "__main__":
    main()
