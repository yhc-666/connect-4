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


class HumanPlayer(Player):
    def __init__(self, player_id=0):
        self.player_id = player_id

    def get_action(self, state):
        """
        In this simple example, choose an action randomly.
        """
        legal_actions = state.legal_actions()

        # 如果没有合法动作，跳过
        if not legal_actions:
            print("没有合法动作，游戏结束!")
            return

        # 打印列号以便输入
        print("可用列 (0-6):", legal_actions)

        # 获取人类输入
        while True:
            try:
                action = int(input("输入你的选择 (0-6): "))
                if action in legal_actions:
                    break
                else:
                    print("无效的动作，请从合法动作中选择")
            except ValueError:
                print("请输入一个有效的数字")
        return action
