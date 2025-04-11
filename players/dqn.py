from dqn import DQNAgent
from players.player import Player


class DQNPlayer(Player):
    def __init__(self, input_shape, action_size, model_path, mcts=None, game=None):
        agent = DQNAgent(input_shape, action_size)
        agent.load(model_path)
        self.agent = agent
        self.mcts = mcts

    def get_action(self, state):
        # TODO: if self.mcts:
        return self.agent.select_action(state)
