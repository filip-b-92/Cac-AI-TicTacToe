"""
Comprehensive Model Evaluation

This script evaluates the RL-trained model against various opponents to determine
if the training was successful and worthwhile compared to algorithmic solutions.

Evaluation opponents:
1. Minimax (perfect play) - The gold standard
2. Random player - Baseline performance
3. Another RL model - Self-play performance

Key Questions:
- Did the RL model learn optimal or near-optimal play?
- Can it draw against perfect play?
- Does it consistently beat random players?
- Was RL training worth it vs just using minimax?

Usage:
    python evaluate_model.py
    python evaluate_model.py --model tic_tac_toe_best_model.pth --games 1000
    python evaluate_model.py --quick  # Fast evaluation with fewer games
"""

import torch
import argparse
import random
import numpy as np
from game import TicTacToe
from model import DQN
from minimax_solver import MinimaxPlayer, get_game_complexity_stats


class ModelEvaluator:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = DQN(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ Model loaded: {model_path}")
        except FileNotFoundError:
            print(f"❌ Model not found: {model_path}")
            print("Train a model first with: python main.py")
            raise

    def get_model_action(self, state, available_actions):
        """Get action from RL model"""
        with torch.no_grad():
            q_values = self.model(state)
            masked_q_values = q_values.clone()
            for i in range(9):
                if i not in available_actions:
                    masked_q_values[i] = -float('inf')
            action = torch.argmax(masked_q_values).item()
        return action

    def evaluate_vs_minimax(self, num_games=100, model_plays_first=True):
        """
        Evaluate RL model against perfect minimax player

        Returns:
            (wins, losses, draws) from model's perspective
        """
        minimax = MinimaxPlayer(player=-1 if model_plays_first else 1)
        model_player = 1 if model_plays_first else -1

        wins = 0
        losses = 0
        draws = 0

        for _ in range(num_games):
            game = TicTacToe()
            state = torch.FloatTensor(game.reset()).to(self.device)
            done = False

            while not done:
                if model_plays_first:
                    # Model's turn (Player 1)
                    action = self.get_model_action(state, game.available_actions())
                    next_state_np, reward, done = game.step(action, model_player)
                    state = torch.FloatTensor(next_state_np).to(self.device)

                    if done:
                        if reward == 1:
                            wins += 1
                        else:
                            draws += 1
                        break

                    # Minimax's turn (Player -1)
                    action = minimax.get_action(game)
                    next_state_np, reward, done = game.step(action, -model_player)
                    state = torch.FloatTensor(next_state_np).to(self.device)

                    if done:
                        if reward != 0:
                            losses += 1
                        else:
                            draws += 1
                else:
                    # Minimax's turn (Player 1)
                    action = minimax.get_action(game)
                    next_state_np, reward, done = game.step(action, -model_player)
                    state = torch.FloatTensor(next_state_np).to(self.device)

                    if done:
                        if reward != 0:
                            losses += 1
                        else:
                            draws += 1
                        break

                    # Model's turn (Player -1)
                    action = self.get_model_action(state, game.available_actions())
                    next_state_np, reward, done = game.step(action, model_player)
                    state = torch.FloatTensor(next_state_np).to(self.device)

                    if done:
                        if reward == 1:
                            wins += 1
                        else:
                            draws += 1

        return wins, losses, draws

    def evaluate_vs_random(self, num_games=100):
        """Evaluate RL model against random player"""
        wins = 0
        losses = 0
        draws = 0

        for _ in range(num_games):
            game = TicTacToe()
            state = torch.FloatTensor(game.reset()).to(self.device)
            done = False

            while not done:
                # Model's turn
                action = self.get_model_action(state, game.available_actions())
                next_state_np, reward, done = game.step(action, 1)
                state = torch.FloatTensor(next_state_np).to(self.device)

                if done:
                    if reward == 1:
                        wins += 1
                    else:
                        draws += 1
                    break

                # Random opponent's turn
                action = random.choice(game.available_actions())
                next_state_np, reward, done = game.step(action, -1)
                state = torch.FloatTensor(next_state_np).to(self.device)

                if done:
                    if reward != 0:
                        losses += 1
                    else:
                        draws += 1

        return wins, losses, draws

    def comprehensive_evaluation(self, games_per_test=100):
        """Run comprehensive evaluation suite"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)

        # Show game complexity
        stats = get_game_complexity_stats()
        print("\n📊 Tic-Tac-Toe Game Complexity:")
        print(f"  Total possible states: {stats['total_possible_states']:,}")
        print(f"  Legal positions: ~{stats['legal_positions_estimate']:,}")
        print(f"  Result with perfect play: {stats['result_with_perfect_play']}")
        print(f"\n  ⚠️  A perfect player will NEVER lose, only draw or win!")

        # Test 1: Model vs Minimax (Model plays first)
        print("\n" + "=" * 70)
        print(f"TEST 1: RL Model vs Minimax (Perfect Play) - {games_per_test} games")
        print("Model plays as X (first), Minimax plays as O (second)")
        print("=" * 70)

        wins, losses, draws = self.evaluate_vs_minimax(games_per_test, model_plays_first=True)
        win_rate = wins / games_per_test * 100
        loss_rate = losses / games_per_test * 100
        draw_rate = draws / games_per_test * 100

        print(f"\nResults:")
        print(f"  Model wins: {wins} ({win_rate:.1f}%)")
        print(f"  Model losses: {losses} ({loss_rate:.1f}%)")
        print(f"  Draws: {draws} ({draw_rate:.1f}%)")

        print(f"\n📈 Analysis:")
        if losses == 0:
            print(f"  ✅ EXCELLENT! Model never lost to perfect play")
            if draw_rate >= 95:
                print(f"  ✅ OPTIMAL! Model achieved {draw_rate:.1f}% draws - near-perfect play!")
            elif draw_rate >= 80:
                print(f"  ✅ VERY GOOD! Model achieved {draw_rate:.1f}% draws")
            else:
                print(f"  ⚠️  Model only drew {draw_rate:.1f}% - still learning optimal play")
        else:
            print(f"  ❌ Model lost {losses} games - should NEVER lose to perfect opponent")
            print(f"  ❌ Model has not learned optimal strategy")

        # Test 2: Model vs Minimax (Model plays second)
        print("\n" + "=" * 70)
        print(f"TEST 2: RL Model vs Minimax (Perfect Play) - {games_per_test} games")
        print("Minimax plays as X (first), Model plays as O (second)")
        print("=" * 70)

        wins2, losses2, draws2 = self.evaluate_vs_minimax(games_per_test, model_plays_first=False)
        win_rate2 = wins2 / games_per_test * 100
        loss_rate2 = losses2 / games_per_test * 100
        draw_rate2 = draws2 / games_per_test * 100

        print(f"\nResults:")
        print(f"  Model wins: {wins2} ({win_rate2:.1f}%)")
        print(f"  Model losses: {losses2} ({loss_rate2:.1f}%)")
        print(f"  Draws: {draws2} ({draw_rate2:.1f}%)")

        print(f"\n📈 Analysis:")
        if losses2 == 0:
            print(f"  ✅ EXCELLENT! Model never lost to perfect play")
            if draw_rate2 >= 95:
                print(f"  ✅ OPTIMAL! Model achieved {draw_rate2:.1f}% draws - near-perfect play!")
            elif draw_rate2 >= 80:
                print(f"  ✅ VERY GOOD! Model achieved {draw_rate2:.1f}% draws")
            else:
                print(f"  ⚠️  Model only drew {draw_rate2:.1f}% - still learning optimal play")
        else:
            print(f"  ❌ Model lost {losses2} games - should NEVER lose to perfect opponent")

        # Test 3: Model vs Random
        print("\n" + "=" * 70)
        print(f"TEST 3: RL Model vs Random Player - {games_per_test} games")
        print("=" * 70)

        wins3, losses3, draws3 = self.evaluate_vs_random(games_per_test)
        win_rate3 = wins3 / games_per_test * 100
        loss_rate3 = losses3 / games_per_test * 100
        draw_rate3 = draws3 / games_per_test * 100

        print(f"\nResults:")
        print(f"  Model wins: {wins3} ({win_rate3:.1f}%)")
        print(f"  Model losses: {losses3} ({loss_rate3:.1f}%)")
        print(f"  Draws: {draws3} ({draw_rate3:.1f}%)")

        print(f"\n📈 Analysis:")
        if win_rate3 >= 95:
            print(f"  ✅ EXCELLENT! {win_rate3:.1f}% win rate vs random - model is very strong")
        elif win_rate3 >= 85:
            print(f"  ✅ GOOD! {win_rate3:.1f}% win rate vs random")
        elif win_rate3 >= 70:
            print(f"  ⚠️  {win_rate3:.1f}% win rate vs random - could be better")
        else:
            print(f"  ❌ Only {win_rate3:.1f}% win rate vs random - model needs more training")

        # Final verdict
        print("\n" + "=" * 70)
        print("FINAL VERDICT: Was RL Training Worth It?")
        print("=" * 70)

        total_losses = losses + losses2
        avg_draw_rate_vs_perfect = (draw_rate + draw_rate2) / 2

        if total_losses == 0 and avg_draw_rate_vs_perfect >= 90 and win_rate3 >= 90:
            print("\n🏆 VERDICT: RL TRAINING WAS HIGHLY SUCCESSFUL!")
            print("  ✅ Model learned near-optimal or optimal play")
            print("  ✅ Never loses to perfect opponent")
            print("  ✅ Dominates random players")
            print(f"  ✅ Achieves ~{avg_draw_rate_vs_perfect:.0f}% draws vs perfect play")
            print("\n  💡 However, consider that a simple minimax algorithm:")
            print("     - Requires no training (instant)")
            print("     - Uses no memory for model weights")
            print("     - Guarantees perfect play")
            print("     - Is computationally efficient for this problem")
            print("\n  🎯 RL was a good learning exercise, but minimax is simpler!")

        elif total_losses == 0 and avg_draw_rate_vs_perfect >= 70:
            print("\n✅ VERDICT: RL TRAINING WAS SUCCESSFUL!")
            print("  ✅ Model learned strong play (never loses to perfect opponent)")
            print(f"  ⚠️  Draw rate vs perfect: {avg_draw_rate_vs_perfect:.0f}% (could be higher)")
            print("\n  💡 Model is good but not optimal yet")
            print("     Consider: More training episodes or hyperparameter tuning")

        elif total_losses <= 10:
            print("\n⚠️  VERDICT: RL TRAINING WAS PARTIALLY SUCCESSFUL")
            print(f"  ⚠️  Model lost {total_losses} games to perfect opponent")
            print("  ⚠️  Model has not fully learned optimal strategy")
            print("\n  💡 Recommendations:")
            print("     - Train for more episodes")
            print("     - Adjust hyperparameters (epsilon decay, learning rate)")
            print("     - Use minimax algorithm instead")

        else:
            print("\n❌ VERDICT: RL TRAINING FAILED")
            print(f"  ❌ Model lost {total_losses} games to perfect opponent")
            print("  ❌ Model has not learned effective strategy")
            print("\n  💡 Recommendations:")
            print("     - Check training implementation for bugs")
            print("     - Review reward structure")
            print("     - Use minimax algorithm instead of RL")

        print("\n" + "=" * 70)
        print("COMPARISON: RL vs Minimax Algorithm")
        print("=" * 70)
        print("\nRL Model:")
        print(f"  + Performance: {avg_draw_rate_vs_perfect:.0f}% draws vs perfect play")
        print(f"  + Training time: Minutes to hours")
        print(f"  + Memory: ~100KB model weights")
        print(f"  + Deployment: Requires PyTorch")
        print(f"  - Needs training")
        print(f"  - Not guaranteed perfect")

        print("\nMinimax Algorithm:")
        print(f"  + Performance: 100% draws vs perfect play (provably optimal)")
        print(f"  + Training time: Zero (no training needed)")
        print(f"  + Memory: Minimal (just code)")
        print(f"  + Deployment: Pure Python, no dependencies")
        print(f"  + Instant deployment")
        print(f"  + Guaranteed perfect play")

        print("\n🎯 For Tic-Tac-Toe specifically: Minimax is the clear winner!")
        print("   However, RL is essential for complex games where minimax is too slow.")

        return {
            'vs_minimax_first': (wins, losses, draws),
            'vs_minimax_second': (wins2, losses2, draws2),
            'vs_random': (wins3, losses3, draws3),
            'total_losses_vs_perfect': total_losses,
            'avg_draw_rate_vs_perfect': avg_draw_rate_vs_perfect
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL model vs algorithmic solutions')
    parser.add_argument('--model', type=str, default='tic_tac_toe_best_model.pth',
                       help='Path to model file')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games per test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation (50 games per test)')

    args = parser.parse_args()

    games = 50 if args.quick else args.games

    try:
        evaluator = ModelEvaluator(args.model)
        results = evaluator.comprehensive_evaluation(games_per_test=games)

        print("\n✅ Evaluation complete!")

    except FileNotFoundError:
        print("\n💡 Train a model first with: python main.py")
        return
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
