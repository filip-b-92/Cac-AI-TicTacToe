"""
Minimax Algorithm - Perfect Tic-Tac-Toe Player

This implements the optimal strategy for Tic-Tac-Toe using the minimax algorithm.
A perfect player using minimax will NEVER lose - only win or draw.

Tic-Tac-Toe Complexity:
- Total possible board states: 3^9 = 19,683
- Legal game positions: ~5,478
- Unique positions (after symmetry): ~765
- Game tree complexity: ~255,168 nodes
- First player (X) can force a draw with perfect play
- Second player (O) can force a draw with perfect play
- Result with perfect play on both sides: ALWAYS DRAW

This provides ground truth to verify if RL training actually worked.
"""

import numpy as np
from game import TicTacToe


class MinimaxPlayer:
    def __init__(self, player=1):
        """
        Initialize minimax player

        Args:
            player: 1 for X (maximizing), -1 for O (minimizing)
        """
        self.player = player
        self.move_count = 0
        self.cache = {}  # Memoization for visited positions

    def get_action(self, game):
        """Get the best action using minimax algorithm"""
        self.move_count += 1
        available = game.available_actions()

        if len(available) == 0:
            return None

        # If only one move available, return it immediately
        if len(available) == 1:
            return available[0]

        best_score = -float('inf') if self.player == 1 else float('inf')
        best_action = available[0]

        for action in available:
            # Simulate the move
            game_copy = self._copy_game(game)
            game_copy.step(action, self.player)

            # Get the score for this move
            score = self._minimax(game_copy, 0, -float('inf'), float('inf'), self.player == -1)

            # Update best move
            if self.player == 1:  # Maximizing player
                if score > best_score:
                    best_score = score
                    best_action = action
            else:  # Minimizing player
                if score < best_score:
                    best_score = score
                    best_action = action

        return best_action

    def _minimax(self, game, depth, alpha, beta, is_minimizing):
        """
        Minimax algorithm with alpha-beta pruning

        Args:
            game: Current game state
            depth: Current depth in game tree
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_minimizing: True if current player is minimizing

        Returns:
            Best score for current position
        """
        # Check cache
        board_key = tuple(game.board)
        if board_key in self.cache:
            return self.cache[board_key]

        # Terminal state check
        if game.done:
            if game.winner == 1:  # X wins
                score = 10 - depth  # Prefer faster wins
            elif game.winner == -1:  # O wins
                score = -10 + depth  # Prefer faster wins
            else:  # Draw
                score = 0
            self.cache[board_key] = score
            return score

        available = game.available_actions()

        if is_minimizing:
            min_score = float('inf')
            for action in available:
                game_copy = self._copy_game(game)
                game_copy.step(action, -1)
                score = self._minimax(game_copy, depth + 1, alpha, beta, False)
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            self.cache[board_key] = min_score
            return min_score
        else:
            max_score = -float('inf')
            for action in available:
                game_copy = self._copy_game(game)
                game_copy.step(action, 1)
                score = self._minimax(game_copy, depth + 1, alpha, beta, True)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            self.cache[board_key] = max_score
            return max_score

    def _copy_game(self, game):
        """Create a deep copy of the game state"""
        new_game = TicTacToe()
        new_game.board = game.board.copy()
        new_game.done = game.done
        new_game.winner = game.winner
        return new_game

    def reset_cache(self):
        """Clear the memoization cache"""
        self.cache = {}


def get_game_complexity_stats():
    """
    Calculate and return statistics about Tic-Tac-Toe game complexity
    """
    stats = {
        'total_possible_states': 3**9,  # 19,683
        'legal_positions_estimate': 5478,
        'unique_positions_symmetry': 765,
        'game_tree_nodes': 255168,
        'optimal_first_move_choices': 3,  # Corner, edge, or center (due to symmetry)
        'max_game_length': 9,
        'min_moves_to_win': 5,
        'result_with_perfect_play': 'DRAW',
    }
    return stats


def test_minimax_perfect_play():
    """
    Test that minimax player never loses
    Returns statistics about perfect play
    """
    print("=" * 70)
    print("Testing Minimax Algorithm - Perfect Play Verification")
    print("=" * 70)

    stats = get_game_complexity_stats()
    print("\n📊 Tic-Tac-Toe Complexity:")
    print(f"  Total possible states: {stats['total_possible_states']:,}")
    print(f"  Legal game positions: ~{stats['legal_positions_estimate']:,}")
    print(f"  Unique positions (symmetry): ~{stats['unique_positions_symmetry']}")
    print(f"  Game tree nodes: ~{stats['game_tree_nodes']:,}")
    print(f"  Result with perfect play: {stats['result_with_perfect_play']}")

    print("\n🎮 Testing: Minimax vs Minimax (100 games)")
    print("Expected: 100% draws\n")

    player1 = MinimaxPlayer(player=1)
    player2 = MinimaxPlayer(player=-1)

    games = 100
    draws = 0
    p1_wins = 0
    p2_wins = 0

    for i in range(games):
        game = TicTacToe()
        game.reset()
        done = False

        while not done:
            # Player 1's turn
            action = player1.get_action(game)
            _, _, done = game.step(action, 1)

            if done:
                break

            # Player 2's turn
            action = player2.get_action(game)
            _, _, done = game.step(action, -1)

        if game.winner == 1:
            p1_wins += 1
        elif game.winner == -1:
            p2_wins += 1
        else:
            draws += 1

    print(f"Results:")
    print(f"  Player 1 (X) wins: {p1_wins}")
    print(f"  Player 2 (O) wins: {p2_wins}")
    print(f"  Draws: {draws}")

    if draws == games:
        print(f"\n✅ PERFECT! Minimax is working correctly - all games were draws")
    else:
        print(f"\n❌ ERROR! Minimax implementation has bugs")

    return draws == games


if __name__ == "__main__":
    test_minimax_perfect_play()
