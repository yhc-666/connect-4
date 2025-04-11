import matplotlib as plt
from players.minimax import MinimaxPlayer
from players.player import Player
from utils import get_connect_four_game, get_initial_state, play_game, visualize_board
import warnings


def play(player1: Player, player2: Player, visualize=False) -> int:
    """Simulates a game between two agents
    Returns integer representing outcome from perspective of `agent1`:
    1 if `agent1` wins
    -1 if `agent1` loses
    0 if draw
    """
    game = get_connect_four_game()
    input_shape = (3, 6, 7)  # [通道, 行, 列]
    action_size = 7  # Connect4有7列可以下子
    state = get_initial_state()

    while not state.is_terminal():
        if visualize:
            print("\n当前棋盘状态:")
            print(state)
            visualize_board(state)

        current_player = state.current_player()
        current_player = player1 if current_player == 0 else player2

        #     # 使用MCTS搜索
        #     best_action, _, action_counts, _ = current_mcts.search(state)
        #     action = best_action
        # else:
        #     # 仅使用DQN策略
        #     state_repr = current_mcts.get_state_representation(state, current_player)
        #     action = current_agent.select_action(state_repr)
        action = current_player.get_action(state)
        # print(f"智能体 {current_player+1} 选择列 {action}")
        state.apply_action(action)

    if visualize:
        print("\n最终棋盘状态:")
        print(state)
        visualize_board(state)

    returns = state.returns()
    if returns[0] > 0:
        print("智能体1赢了!")
        return 1
    elif returns[1] > 0:
        print("智能体2赢了!")
        return -1
    else:
        print("平局!")
        return 0


def get_winrate_against_minimax(player: Player, depth=2, num_games=50) -> float:
    """Plays 50 games against minimax agent,
    returns winrate with 1.0 representing all games won"""
    if num_games % 2 != 0:
        warnings.warn(
            f"Number of games {num_games} indicated is not even! Evaluation might be unfair."
        )
    game = get_connect_four_game()
    player.set_game(game)
    games_won = 0
    total_games = num_games
    minimax = MinimaxPlayer(player_id=0, game=game, max_depth=depth)
    player.set_player_id(1)
    for _ in range(total_games // 2):
        games_won += play(player, minimax)
    minimax = MinimaxPlayer(player_id=1, game=game, max_depth=depth)
    player.set_player_id(0)
    for _ in range(total_games - total_games // 2):
        games_won += play(player, minimax)
    return games_won / total_games


def plot_winrate(winrates):

    pass
