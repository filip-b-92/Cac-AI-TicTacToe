"""
Model Inspection and Visualization Utility

This script provides tools to visualize and understand what the trained model is thinking:
- Visualize Q-values for any board position
- Show model's top action choices
- Compare Q-values across different board states
- Analyze model decision-making

Usage:
    python inspect_model.py --board "X..O....." --model tic_tac_toe_best_model.pth
    python inspect_model.py --interactive
"""

import torch
import numpy as np
import argparse
from model import DQN
from game import TicTacToe


class ModelInspector:
    def __init__(self, model_path='tic_tac_toe_best_model.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = DQN(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {self.device}\n")

    def board_to_state(self, board_str):
        """Convert a board string like 'X..O.....' to a state tensor
        X = 1 (AI), O = -1 (opponent), . = 0 (empty)
        """
        mapping = {'X': 1, 'O': -1, '.': 0, ' ': 0}
        board = np.array([mapping.get(c, 0) for c in board_str])
        return torch.FloatTensor(board).to(self.device)

    def state_to_board_str(self, state):
        """Convert state tensor to readable string"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        mapping = {1: 'X', -1: 'O', 0: '.'}
        return ''.join([mapping[int(x)] for x in state])

    def print_board(self, board):
        """Pretty print a tic-tac-toe board"""
        if isinstance(board, str):
            symbols = list(board)
        else:
            if isinstance(board, torch.Tensor):
                board = board.cpu().numpy()
            mapping = {1: 'X', -1: 'O', 0: '.'}
            symbols = [mapping[int(x)] for x in board]

        print(f" {symbols[0]} | {symbols[1]} | {symbols[2]} ")
        print("-----------")
        print(f" {symbols[3]} | {symbols[4]} | {symbols[5]} ")
        print("-----------")
        print(f" {symbols[6]} | {symbols[7]} | {symbols[8]} ")

    def get_q_values(self, state):
        """Get Q-values for a given state"""
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.cpu().numpy()

    def visualize_q_values(self, board_str, show_heatmap=True):
        """Visualize Q-values for a board position"""
        state = self.board_to_state(board_str)
        q_values = self.get_q_values(state)

        # Get available actions
        available = [i for i, c in enumerate(board_str) if c in ['.', ' ']]

        print("=" * 50)
        print("Board State:")
        print("=" * 50)
        self.print_board(board_str)
        print()

        print("=" * 50)
        print("Q-Values for Each Position:")
        print("=" * 50)
        print(f" {q_values[0]:7.3f} | {q_values[1]:7.3f} | {q_values[2]:7.3f} ")
        print("-" * 35)
        print(f" {q_values[3]:7.3f} | {q_values[4]:7.3f} | {q_values[5]:7.3f} ")
        print("-" * 35)
        print(f" {q_values[6]:7.3f} | {q_values[7]:7.3f} | {q_values[8]:7.3f} ")
        print()

        # Show top actions
        valid_q = [(i, q_values[i]) for i in available]
        valid_q.sort(key=lambda x: x[1], reverse=True)

        print("=" * 50)
        print("Model's Action Ranking (for valid moves):")
        print("=" * 50)
        position_names = [
            "Top-Left", "Top-Center", "Top-Right",
            "Mid-Left", "Center", "Mid-Right",
            "Bot-Left", "Bot-Center", "Bot-Right"
        ]

        for rank, (pos, q_val) in enumerate(valid_q, 1):
            marker = "👉 BEST" if rank == 1 else ""
            print(f"{rank}. Position {pos} ({position_names[pos]}): Q={q_val:.4f} {marker}")

        print()

        # Show if there are any occupied positions
        occupied = [i for i in range(9) if i not in available]
        if occupied:
            print("Occupied positions (invalid moves):")
            for pos in occupied:
                print(f"  Position {pos} ({position_names[pos]}): Q={q_values[pos]:.4f} (invalid)")
            print()

    def compare_positions(self, board_strings):
        """Compare Q-values across multiple board positions"""
        print("=" * 50)
        print("Comparing Multiple Board Positions")
        print("=" * 50)

        for i, board_str in enumerate(board_strings, 1):
            print(f"\n--- Position {i} ---")
            state = self.board_to_state(board_str)
            q_values = self.get_q_values(state)
            self.print_board(board_str)

            available = [j for j, c in enumerate(board_str) if c in ['.', ' ']]
            if available:
                best_action = max(available, key=lambda x: q_values[x])
                print(f"Best action: Position {best_action}, Q-value: {q_values[best_action]:.4f}")
            print()

    def analyze_game_scenario(self, description, board_str):
        """Analyze a specific game scenario"""
        print("\n" + "=" * 50)
        print(f"Scenario: {description}")
        print("=" * 50)
        self.visualize_q_values(board_str, show_heatmap=False)

    def interactive_mode(self):
        """Interactive mode for exploring model decisions"""
        print("\n" + "=" * 50)
        print("Interactive Model Inspector")
        print("=" * 50)
        print("Enter a board state using X, O, and . (dot for empty)")
        print("Example: 'X..O.....' for X in top-left, O in position 3")
        print("Positions are numbered 0-8, left to right, top to bottom")
        print("Type 'quit' to exit\n")

        while True:
            board_str = input("Enter board state (or 'quit'): ").strip()

            if board_str.lower() in ['quit', 'exit', 'q']:
                break

            if len(board_str) != 9:
                print("❌ Error: Board must be exactly 9 characters")
                continue

            if not all(c in 'XxOo. ' for c in board_str):
                print("❌ Error: Use only X, O, and . (or space for empty)")
                continue

            board_str = board_str.upper().replace(' ', '.')

            try:
                self.visualize_q_values(board_str)
            except Exception as e:
                print(f"❌ Error: {e}")

        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Inspect and visualize Tic-Tac-Toe AI model')
    parser.add_argument('--model', type=str, default='tic_tac_toe_best_model.pth',
                       help='Path to model file')
    parser.add_argument('--board', type=str, default=None,
                       help='Board state to analyze (e.g., "X..O.....")')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')
    parser.add_argument('--examples', action='store_true',
                       help='Show example scenarios')

    args = parser.parse_args()

    try:
        inspector = ModelInspector(args.model)
    except FileNotFoundError:
        print(f"❌ Error: Model file '{args.model}' not found")
        print("Make sure you've trained a model first!")
        return

    if args.interactive:
        inspector.interactive_mode()
    elif args.examples:
        print("\n📚 Example Scenarios:\n")

        inspector.analyze_game_scenario(
            "Empty board - Opening move",
            "........."
        )

        inspector.analyze_game_scenario(
            "Can win in one move (horizontal)",
            "XX.O..O.."
        )

        inspector.analyze_game_scenario(
            "Must block opponent's win",
            "OO.X..X.."
        )

        inspector.analyze_game_scenario(
            "Fork opportunity",
            "X...O...X"
        )

    elif args.board:
        inspector.visualize_q_values(args.board)
    else:
        print("Usage examples:")
        print("  python inspect_model.py --interactive")
        print("  python inspect_model.py --examples")
        print("  python inspect_model.py --board 'X..O.....'")
        print("\nUse --help for more options")


if __name__ == "__main__":
    main()
