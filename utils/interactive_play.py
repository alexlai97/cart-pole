"""
Interactive Cart-Pole Player Module

Consolidated interactive play functionality for Cart-Pole environment.
Supports both turn-based and real-time gameplay modes.
"""


import gymnasium as gym
import pygame


class CartPolePlayer:
    """Interactive Cart-Pole player with multiple control modes."""

    def __init__(self, render_mode: str = "human"):
        """Initialize the Cart-Pole player."""
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.episode_count = 0
        self.best_score = 0

    def play_turn_based(self) -> None:
        """Play Cart-Pole with turn-based controls (think each move)."""
        print("🎯 Turn-based Cart-Pole Control")
        print("=" * 40)
        print("Controls: Type 'l' for left, 'r' for right, 'q' to quit")
        print("After each action, press Enter")
        print("Goal: Keep the pole balanced as long as possible!")

        state, _ = self.env.reset()
        step_count = 0

        while True:
            self.env.render()

            print(f"\nStep {step_count + 1}")
            print(f"State: pos={state[0]:.2f}, vel={state[1]:.2f}, "
                  f"angle={state[2]:.2f}, ang_vel={state[3]:.2f}")

            user_input = input("Action (l/r/q): ").strip().lower()

            if user_input == 'q':
                break
            elif user_input == 'l':
                action = 0  # Push left
            elif user_input == 'r':
                action = 1  # Push right
            else:
                print("Invalid input, using right")
                action = 1

            state, reward, terminated, truncated, _ = self.env.step(action)
            step_count += 1

            if terminated or truncated:
                self._update_scores(step_count)
                print(f"🏁 Episode ended after {step_count} steps!")

                reset = input("Reset? (y/n): ").strip().lower()
                if reset == 'y':
                    state, _ = self.env.reset()
                    step_count = 0
                else:
                    break

        self.env.close()

    def play_realtime(self) -> None:
        """Play Cart-Pole with real-time controls (continuous A/D keys)."""
        print("🎮 Real-time Cart-Pole Controller")
        print("=" * 40)
        print("Controls (focus on game window):")
        print("  A or LEFT  - Push cart left (hold for continuous)")
        print("  D or RIGHT - Push cart right (hold for continuous)")
        print("  R          - Reset environment")
        print("  ESC        - Quit")
        print()
        print("🎯 Goal: Keep the pole balanced as long as possible!")
        print("The game runs in real-time - pole will fall if you don't act!")

        input("\nPress Enter to start (then focus on the game window)...")

        # Initialize pygame for input handling
        pygame.init()
        clock = pygame.time.Clock()

        state, _ = self.env.reset()
        running = True
        episode_step = 0
        last_action = 1  # Default to right

        print("🚀 Game started! Focus on Cart-Pole window and use A/D keys")

        while running:
            current_action = None

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        state, _ = self.env.reset()
                        episode_step = 0
                        print(f"🔄 Reset! Episode {self.episode_count + 1} starting...")

            # Check continuous key states
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                current_action = 0  # Push left
                last_action = 0
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                current_action = 1  # Push right
                last_action = 1
            else:
                # Maintain last action for momentum
                current_action = last_action

            # Take action in environment
            if current_action is not None:
                state, reward, terminated, truncated, _ = self.env.step(current_action)
                episode_step += 1

                # Optional progress info (every 30 steps)
                if episode_step % 30 == 0:
                    angle_deg = state[2] * 180 / 3.14159
                    print(f"Step {episode_step}: angle={angle_deg:.1f}°, pos={state[0]:.2f}")

                # Check if episode ended
                if terminated or truncated:
                    self._update_scores(episode_step)
                    print(f"📊 Episode {self.episode_count}: {episode_step} steps "
                          f"(Best: {self.best_score})")
                    print("Press 'R' to reset or ESC to quit")
                    episode_step = 0

            # Render and control frame rate
            self.env.render()
            clock.tick(60)  # 60 FPS

        # Cleanup
        pygame.quit()
        self.env.close()

        print("\n📊 Final Statistics:")
        print(f"  Episodes played: {self.episode_count}")
        print(f"  Best performance: {self.best_score} steps")
        print("  Thanks for playing! 🎮")

    def play_with_strategies(self) -> None:
        """Demonstrate different control strategies."""
        print("🧠 Cart-Pole Strategy Demonstrations")
        print("=" * 40)
        print("Watch different control strategies in action")
        input("Press Enter to start...")

        strategies = [
            ("Random Actions", lambda state: self.env.action_space.sample()),
            ("Always Left", lambda state: 0),
            ("Always Right", lambda state: 1),
            ("Simple Balance", lambda state: 0 if state[2] < 0 else 1),
            ("Advanced Balance", lambda state: 0 if state[2] + state[3] * 0.1 < 0 else 1),
        ]

        for strategy_name, strategy_func in strategies:
            print(f"\n🧠 Strategy: {strategy_name}")
            state, _ = self.env.reset()

            for step in range(200):  # Max 200 steps per strategy
                self.env.render()
                action = strategy_func(state)
                state, reward, terminated, truncated, _ = self.env.step(action)

                pygame.time.wait(50)  # Slow down for observation

                if terminated or truncated:
                    print(f"  Lasted {step + 1} steps")
                    break
            else:
                print("  Completed all 200 steps! 🎉")

            pygame.time.wait(1000)  # Pause between strategies

        self.env.close()

    def _update_scores(self, score: int) -> None:
        """Update episode count and best score."""
        self.episode_count += 1
        if score > self.best_score:
            self.best_score = score
            print(f"🏆 NEW BEST! {self.best_score} steps")


def interactive_menu() -> None:
    """Interactive menu for different play modes."""
    while True:
        print("\n" + "="*50)
        print("🎮 Cart-Pole Interactive Play")
        print("="*50)
        print("1. Real-time play (A/D keys - challenging!)")
        print("2. Turn-based play (think each move)")
        print("3. Strategy demonstrations")
        print("4. Exit")

        choice = input("\nChoose mode (1-4): ").strip()

        try:
            if choice == '1':
                player = CartPolePlayer()
                player.play_realtime()
            elif choice == '2':
                player = CartPolePlayer()
                player.play_turn_based()
            elif choice == '3':
                player = CartPolePlayer()
                player.play_with_strategies()
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice, please try again")
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure pygame is installed!")


# Convenience functions for main.py integration
def play_realtime() -> None:
    """Convenience function for real-time play."""
    player = CartPolePlayer()
    player.play_realtime()


def play_turn_based() -> None:
    """Convenience function for turn-based play."""
    player = CartPolePlayer()
    player.play_turn_based()


if __name__ == "__main__":
    interactive_menu()
