from dqn import DQNAgent
from players.player import Player

INPUT_SHAPE = (3, 6, 7)  # [通道, 行, 列]
ACTION_SIZE = 7


class DQNPlayer(Player):
    def __init__(self, model_path, mcts=None, player_id=0):
        agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE)
        agent.load(model_path)
        self.agent = agent
        self.mcts = mcts
        self.player_id = player_id

    def get_action(self, state):
        action = None
        if self.mcts is not None:
            action, _, _, _ = self.mcts.search(state)
        else:
            state_repr = self.mcts.get_state_representation(
                state, self.player_id
            )  # FIXME: Best to remove this function from MCTS class
            action = self.agent.select_action(state_repr)
        return action
