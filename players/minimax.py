from players.player import Player
from open_spiel.python.algorithms.minimax import alpha_beta_search
import numpy as np


class MinimaxPlayer(Player):
    """
    A simple minimax agent using the OpenSpiel Python API.
    """

    def __init__(self, player_id=0, game=None, max_depth=2):
        """
        Parameters:
            max_depth: The maximum search depth for minimax.
        """
        self.player_id = player_id
        self.opponent_id = 1 if player_id == 0 else 0
        self.game = game
        self.max_depth = max_depth

    def get_action(self, state):
        """
        Returns an action by performing a negamax search on the given state.

        Parameters:
            state: An OpenSpiel game state.

        Returns:
            The best action as determined by the search.
        """
        _, action = alpha_beta_search(
            self.game,
            state=state,
            value_function=self._evaluate_board,
            maximum_depth=self.max_depth,
        )
        return action

    def _evaluate_board(self, state) -> float:
        """Heuristic evaluation function for non-terminal board states.

        This function evaluates the board state based on the number of potential
        winning lines for each player. A potential winning line is a line of 4
        cells that could potentially form a winning line (i.e., it doesn't contain
        any opponent's pieces).

        Args:
            board: Current board state.

        Returns:
            A score representing how good the board state is for the player.
            Positive values are good for the player, negative values are good
            for the opponent.
        """
        score = 0
        board = np.array(state.observation_tensor(self.player_id)).reshape(3, 6, 7)[0]
        # Check horizontal windows
        for row in range(6):
            for col in range(4):
                window = board[row, col : col + 4]
                score += self._evaluate_window(window)

        # Check vertical windows
        for row in range(3):
            for col in range(7):
                window = board[row : row + 4, col]
                score += self._evaluate_window(window)

        # Check diagonal windows (positive slope)
        for row in range(3, 6):
            for col in range(4):
                window = [board[row - i, col + i] for i in range(4)]
                score += self._evaluate_window(window)

        # Check diagonal windows (negative slope)
        for row in range(3):
            for col in range(4):
                window = [board[row + i, col + i] for i in range(4)]
                score += self._evaluate_window(window)

        # Prefer center column
        center_col = board[:, 3]
        score += np.sum(center_col == self.player_id) * 3

        return score

    def _evaluate_window(self, window: np.ndarray) -> float:
        """Evaluate a window of 4 cells.

        Args:
            window: A window of 4 cells.

        Returns:
            A score for the window.
        """
        player_count = np.sum(window == self.player_id)
        opponent_count = np.sum(window == self.opponent_id)
        empty_count = np.sum(window == 0)

        # If the window contains both player and opponent pieces, it's not a potential win
        if player_count > 0 and opponent_count > 0:
            return 0

        # Score based on the number of player pieces in the window
        if player_count == 4:
            return 100  # Player wins
        elif player_count == 3 and empty_count == 1:
            return 5  # Player has 3 in a row with an empty cell
        elif player_count == 2 and empty_count == 2:
            return 2  # Player has 2 in a row with 2 empty cells
        elif player_count == 1 and empty_count == 3:
            return 1  # Player has 1 in a row with 3 empty cells

        # Score based on the number of opponent pieces in the window
        if opponent_count == 4:
            return -100  # Opponent wins
        elif opponent_count == 3 and empty_count == 1:
            return (
                -50
            )  # Opponent has 3 in a row with an empty cell - high priority to block!
        elif opponent_count == 2 and empty_count == 2:
            return -2  # Opponent has 2 in a row with 2 empty cells
        elif opponent_count == 1 and empty_count == 3:
            return -1  # Opponent has 1 in a row with 3 empty cells

        return 0
