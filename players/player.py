from abc import ABC, abstractmethod


class Player(ABC):
    """
    Abstract base class for an agent that, given a state, returns an action.
    """

    @abstractmethod
    def get_action(self, state):
        """
        Given the current state, compute and return an action.

        Parameters:
            state: The current environment or game state. The type and structure
                   depend on the problem domain.

        Returns:
            action: The action chosen based on the current state.
        """
        pass

    def set_game(self, game):
        self.game = game

    def set_player_id(self, pid):
        self.player_id = pid


class RandomPlayer(Player):
    def __init__(self, action_space):
        """
        Parameters:
            action_space: An iterable or list of possible actions.
        """
        self.action_space = action_space

    def get_action(self, state):
        """
        In this simple example, choose an action randomly.
        """
        import random

        return random.choice(self.action_space)
