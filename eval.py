import os
from typing import Optional
import matplotlib.pyplot as plt
from mcts_wrapper import MCTSWrapper
from players.dqn import DQNPlayer
from players.minimax import MinimaxPlayer
from players.player import Player
from utils import get_connect_four_game, get_initial_state, visualize_board
import warnings


def play(player1: Player, player2: Player, visualize=False) -> int:
    """Simulates a game between two agents
    Returns integer representing outcome from perspective of `agent1`:
    1 if `agent1` wins
    -1 if `agent1` loses
    0 if draw
    """
    state = get_initial_state()

    while not state.is_terminal():
        if visualize:
            print("\n当前棋盘状态:")
            print(state)
            visualize_board(state)

        current_player = state.current_player()
        current_player = player1 if current_player == 0 else player2

        action = current_player.get_action(state)
        print(f"智能体 {current_player} 选择列 {action}")
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


def evaluate_all_checkpoints(checkpoint_folder, mcts_simulations=1000):
    # List all files that match a checkpoint extension (e.g. '.ckpt').
    # You may adjust the file pattern if your checkpoint files have a different extension.
    checkpoints = [
        f
        for f in os.listdir(checkpoint_folder)
        if f.endswith(".ckpt") or f.endswith(".pt")
    ]

    checkpoints.sort()
    win_rates = []

    for ckpt in checkpoints:
        ckpt_path = os.path.join(checkpoint_folder, ckpt)

        game = get_connect_four_game()
        dqn = DQNPlayer(ckpt_path, mcts=mcts)
        mcts = MCTSWrapper(
            game=game,
            num_simulations=mcts_simulations,
            uct_c=2.0,
            max_nodes=10000,
            dirichlet_alpha=0.0,
            dirichlet_noise=False,
            solve=True,
        )

        win_rate = get_winrate_against_minimax(dqn)
        win_rates.append(win_rate)
        print(f"Evaluated {ckpt}: win rate = {win_rate:.2f}")

    return checkpoints, win_rates


def plot_win_rates(
    win_rates: Optional[float] = None, checkpoint_folder: Optional[str] = None
):
    if not win_rates and not checkpoint_folder:
        raise Exception(
            "Either winrates or checkpoint folder for evaluation shold be provided"
        )

    if not win_rates:
        win_rates = evaluate_all_checkpoints(
            checkpoint_folder,
        )
        # winrates.save

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(win_rates)), win_rates, marker="o", linestyle="-", color="blue")
    plt.xlabel("(Checkpoint) Index")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Evaluation Across Checkpoints")

    # Optionally, set checkpoint names as x-tick labels.
    # If there are many checkpoints, you might want to rotate the labels.
    # plt.xticks(range(len(win_rates)), checkpoint_names, rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
