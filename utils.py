import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pyspiel
from open_spiel.python.algorithms import mcts
from mcts_wrapper import MCTSWrapper, DQNEvaluator
from agents import MiniMaxAgent
from tqdm import tqdm
from collections import deque
from dataclasses import dataclass
from typing import Optional
import random
import torch


def get_connect_four_game():
    """获取Connect4游戏实例"""
    return pyspiel.load_game("connect_four")


def get_initial_state():
    """获取游戏初始状态"""
    game = get_connect_four_game()
    return game.new_initial_state()


def get_state_representation(state, player_id):
    """将状态转换为网络输入格式

    Args:
        state: 游戏状态
        player_id: 玩家ID

    Returns:
        状态表示为numpy数组，形状为 [通道, 行, 列]
    """
    obs = np.array(state.observation_tensor(player_id))
    # Connect4的观察空间：[cell_states, rows, cols] = [3, 6, 7]
    obs = obs.reshape(3, 6, 7)
    return obs


def play_game(
    agent1,
    agent2,
    epsilon=0.0,
    verbose=False,
    collect_experience=False,
    replay_buffer=None,
    lambda_mix=0.5,
    n_step=1,
    gamma=0.99
):
    """进行一局游戏，可选择收集经验到回放缓冲区

    Args:
        agent1: 玩家1的智能体
        agent2: 玩家2的智能体
        epsilon: 探索概率
        verbose: 是否输出游戏过程
        collect_experience: 是否收集经验
        replay_buffer: 经验回放缓冲区
        lambda_mix: MCTS和DQN目标的混合系数（仅在collect_experience=True时使用）

    Returns:
        游戏奖励和终局状态
    """
    state = get_initial_state()
    agents = [agent1, agent2]
    player_experiences = {0: [], 1: []}
    done = False
    length = 1

    while not done:
        current_player = state.current_player()
        agent = agents[current_player]

        # 选择动作
        action = agent.select_action(state, epsilon)
        length += 1

        # 应用选择的动作
        next_state = state.clone()
        next_state.apply_action(action)

        # 计算奖励
        if next_state.is_terminal():
            returns = next_state.returns()
            reward = returns[current_player] # win
            done = True
        else:
            reward = 0.0
            done = False

        # 收集经验到回放缓冲区（如果需要）
        if (
            collect_experience
            and replay_buffer is not None
            and hasattr(agent, "get_q_values")
        ):
            initial_state_repr = get_state_representation(state, current_player)
            # 对于DQN智能体，需要计算Q值
            mcts_q_value = 0.0  # 默认值
            if hasattr(agent, "mcts_wrapper") and agent.num_simulations > 0:
                # 如果是MCTS-DQN智能体，尝试获取MCTS Q值
                _, action_q_values, _, _ = agent.mcts_wrapper.search(state)
                mcts_q_value = action_q_values.get(action, 0.0)
            player_experiences[current_player].append([
                initial_state_repr,
                action,
                reward,
                mcts_q_value,
                done]
            )

        # 更新游戏状态
        state = next_state

        if verbose:
            print(f"玩家 {current_player} 执行动作 {action}")
            print(state)

    # Add losing sample for opponent
    if collect_experience and replay_buffer is not None:
        losing_player = 1 - current_player
        losing_reward = state.returns()[losing_player]
        player_experiences[losing_player][-1][2] = losing_reward
        player_experiences[losing_player][-1][4] = True
        

    # add actual n_step samples
    for states in list(player_experiences.values()):
        for i in range(len(states)):
            last_idx = min(i+n_step-1, len(states)-1)
            s_0, a_0, _, mcts_q_0, _ = states[i]
            s_n, _, r_n, _, done_n = states[last_idx]

            # Compute discounted sum of rewards
            G = 0
            if r_n != 0: # only terminal state has reward
                discount_pow = last_idx - i
                G += (gamma ** discount_pow) * r_n
                
            replay_buffer.add(s_0, a_0, G, s_n, mcts_q_0, done_n)
        
    if verbose:
        print("win")
        print()
        print(action)
        print(action_q_values.items())
        print(losing_reward)
        print(replay_buffer.buffer[-2])
        print(replay_buffer.buffer[-1])
        # import code; code.interact(local=locals())
        print("============================================")

    # 游戏结束，返回奖励
    returns = state.returns()
    if verbose:
        print(f"游戏结束! 奖励: 玩家0 = {returns[0]}, 玩家1 = {returns[1]}")

    return returns, state, length
    
def evaluate_agent(agent1, agent2, num_games=100):
    """评估智能体的性能

    Args:
        agent1: 要评估的智能体
        agent2: 对手智能体（baseline）
        num_games: 每个位置（先手/后手）的评估游戏数量

    Returns:
        agent1作为先手和后手的胜率、平局率和败率
    """
    # 用于存储结果的字典
    results = {
        "first_player": {"wins": 0, "draws": 0, "losses": 0},
        "second_player": {"wins": 0, "draws": 0, "losses": 0},
    }
    set_seed(1)

    # Agent1作为先手(玩家0)进行num_games次游戏
    print(f"评估Agent1作为先手的{num_games}局游戏...")
    for _ in tqdm(range(num_games), desc="Agent1作为先手"):
        state = get_initial_state()

        # 第一步随机行动
        legal_actions = state.legal_actions()
        random_action = np.random.choice(legal_actions)
        state.apply_action(random_action)

        # 继续游戏直到结束
        while not state.is_terminal():
            # print("\n当前棋盘:")
            # print(state.observation_string(0))
            current_player = state.current_player()
            current_agent = agent2 if current_player == 1 else agent1
            action = current_agent.select_action(state, epsilon=0)
            state.apply_action(action)

        # 记录结果
        returns = state.returns()
        if returns[0] > 0:  # agent1获胜
            results["first_player"]["wins"] += 1
        elif returns[1] > 0:  # agent2获胜
            results["first_player"]["losses"] += 1
        else:  # 平局
            results["first_player"]["draws"] += 1
    # Agent1作为后手(玩家1)进行num_games次游戏
    print(f"评估Agent1作为后手的{num_games}局游戏...")
    for _ in tqdm(range(num_games), desc="Agent1作为后手"):
        state = get_initial_state()

        # 第一步随机行动
        # legal_actions = state.legal_actions()
        # random_action = np.random.choice(legal_actions)

        options = np.arange(7)
        mu = 3.0
        sigma = 1.0
        unnormalized = np.exp(-0.5 * ((options - mu) / sigma) ** 2)
        distribution = unnormalized / np.sum(unnormalized)
        sampled_action = np.random.choice(options, p=distribution)

        state.apply_action(sampled_action)

        # 继续游戏直到结束
        while not state.is_terminal():
            # print("\n当前棋盘:")
            # print(state.observation_string(0))
            current_player = state.current_player()
            current_agent = agent1 if current_player == 1 else agent2
            action = current_agent.select_action(state, epsilon=0)
            state.apply_action(action)

        # 记录结果
        returns = state.returns()
        if returns[1] > 0:  # agent1获胜
            results["second_player"]["wins"] += 1
        elif returns[0] > 0:  # agent2获胜
            results["second_player"]["losses"] += 1
        else:  # 平局
            results["second_player"]["draws"] += 1

    # 计算胜率、平局率和败率
    total_first = num_games
    total_second = num_games

    first_win_rate = results["first_player"]["wins"] / total_first
    first_draw_rate = results["first_player"]["draws"] / total_first
    first_loss_rate = results["first_player"]["losses"] / total_first

    second_win_rate = results["second_player"]["wins"] / total_second
    second_draw_rate = results["second_player"]["draws"] / total_second
    second_loss_rate = results["second_player"]["losses"] / total_second

    # 打印结果
    print("\n评估结果:")
    print(
        f"Agent1作为先手: 胜率={first_win_rate:.2f}, 平局率={first_draw_rate:.2f}, 败率={first_loss_rate:.2f}"
    )
    print(
        f"Agent1作为后手: 胜率={second_win_rate:.2f}, 平局率={second_draw_rate:.2f}, 败率={second_loss_rate:.2f}"
    )

    return {
        "first_player": {
            "win_rate": first_win_rate,
            "draw_rate": first_draw_rate,
            "loss_rate": first_loss_rate,
        },
        "second_player": {
            "win_rate": second_win_rate,
            "draw_rate": second_draw_rate,
            "loss_rate": second_loss_rate,
        },
    }


def visualize_board(state):
    """可视化Connect4棋盘

    Args:
        state: 游戏状态
    """
    # 获取棋盘状态
    board_str = state.observation_string(0)
    rows = board_str.strip().split("\n")

    # 转换为数值数组: 0=空格, 1=玩家1(x), 2=玩家2(o)
    board = np.zeros((6, 7), dtype=int)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell == "x":
                board[i, j] = 1
            elif cell == "o":
                board[i, j] = 2

    # 创建颜色映射: 白色=空, 红色=玩家1, 黄色=玩家2
    cmap = ListedColormap(["white", "red", "yellow"])

    plt.figure(figsize=(7, 6))
    plt.imshow(board, cmap=cmap)
    plt.grid(color="black", linestyle="-", linewidth=1.5)
    plt.xticks(np.arange(7))
    plt.yticks(np.arange(6))
    plt.title("Connect Four")
    plt.tight_layout()
    plt.show()


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_moving_average(values, window=100):
    """计算移动平均值

    Args:
        values: 值列表
        window: 窗口大小

    Returns:
        移动平均值列表
    """
    if not values or len(values) < window:
        return []
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def play_interactive_game(agent):
    """人机对战游戏

    Args:
        agent: 智能体实例
    """
    state = get_initial_state()

    print("欢迎来到Connect4游戏！")
    print("玩家1: AI (o)")
    print("玩家2: 人类 (x)")
    print("输入0-6选择落子的列")

    while not state.is_terminal():
        print("\n当前棋盘:")
        print(state.observation_string(0))

        current_player = state.current_player()

        if current_player == 1:  # 人类玩家
            legal_actions = state.legal_actions()
            while True:
                try:
                    action = int(input("请选择列 (0-6): "))
                    if action in legal_actions:
                        break
                    else:
                        print("无效的动作，该列已满或不存在！")
                except ValueError:
                    print("请输入0-6之间的数字！")
        else:  # AI玩家
            print("AI正在思考...")

            # 使用智能体选择动作
            action = agent.select_action(state, epsilon=0)

            # 如果是MCTS智能体，可以打印搜索信息
            if hasattr(agent, "mcts_wrapper") and agent.num_simulations > 0:
                _, action_q_values, action_counts, _ = agent.mcts_wrapper.search(state)
                # 打印MCTS搜索信息
                print("\nMCTS搜索结果:")
                for a in sorted(action_counts.keys()):
                    print(
                        f"列 {a}: 访问次数 = {action_counts[a]}, Q值 = {action_q_values.get(a, 0):.4f}"
                    )

        # 应用动作
        state.apply_action(action)
        print(f"玩家 {current_player+1} 选择了列 {action}")

    # 游戏结束
    print("\n最终棋盘:")
    print(state.observation_string(0))

    returns = state.returns()
    if returns[0] > 0:
        print("恭喜！你赢了！")
    elif returns[1] > 0:
        print("AI获胜！再接再厉！")
    else:
        print("平局！")


def update_elo_rating(player_rating, opponent_rating, score, k_factor=32):
    """
    更新玩家的ELO评分

    Args:
        player_rating: 玩家当前 Elo 评分
        opponent_rating: 对手当前 Elo 评分
        score: 玩家本局得分 (1 for win, 0.5 for draw, 0 for loss)
        k_factor: Elo K-factor (控制评分变化幅度)

    Returns:
        更新后的玩家 Elo 评分
    """
    expected_score = 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))
    new_rating = player_rating + k_factor * (score - expected_score)
    return new_rating


def evaluate_agent_elo(
    agent, opponent_agent, current_elos, opponent_minimax_depth=7, k_factor=32
):
    """
    使用 Elo 评分评估智能体性能，与固定深度的 Minimax 对战。

    Args:
        agent: 要评估的智能体。
        opponent_agent: 对手智能体 (这里特指 Minimax).
        current_elos: 包含当前 Elo 评分的字典:
                      {'agent_p1_elo': elo, 'agent_p2_elo': elo,
                       'minimax_p1_elo': elo, 'minimax_p2_elo': elo}
        opponent_minimax_depth: Minimax 对手的搜索深度。
        k_factor: Elo K-factor.

    Returns:
        包含更新后 Elo 评分的字典。
    """
    # 获取当前的 Elo 评分
    agent_p1_elo = current_elos["agent_p1_elo"]
    agent_p2_elo = current_elos["agent_p2_elo"]
    minimax_p1_elo = current_elos["minimax_p1_elo"]
    minimax_p2_elo = current_elos["minimax_p2_elo"]

    # 确保对手是 MiniMaxAgent 且深度符合要求
    # Note: We might not need to recreate the opponent if it's passed correctly.
    # Let's assume opponent_agent is already the correct Minimax agent.
    if not isinstance(opponent_agent, MiniMaxAgent):
        print(
            f"Warning: Opponent for Elo evaluation is not MiniMaxAgent, but {type(opponent_agent)}"
        )
        # Or raise an error: raise TypeError("Opponent must be a MiniMaxAgent for Elo evaluation")

    # --- Game 1: Agent (P1) vs Minimax (P2) ---
    print("Elo Evaluation: Game 1 - Agent (P1) vs Minimax (P2)")
    returns_g1, _, _ = play_game(
        agent, opponent_agent, verbose=False, collect_experience=False
    )
    score_agent_g1 = (
        0.5 if returns_g1[0] == returns_g1[1] else (1.0 if returns_g1[0] > 0 else 0.0)
    )
    score_minimax_g1 = 1.0 - score_agent_g1

    # 更新 Elo (Agent P1 vs Minimax P2)
    new_agent_p1_elo = update_elo_rating(
        agent_p1_elo, minimax_p2_elo, score_agent_g1, k_factor
    )
    new_minimax_p2_elo = update_elo_rating(
        minimax_p2_elo, agent_p1_elo, score_minimax_g1, k_factor
    )

    # --- Game 2: Minimax (P1) vs Agent (P2) ---
    print("Elo Evaluation: Game 2 - Minimax (P1) vs Agent (P2)")
    returns_g2, _, _ = play_game(
        opponent_agent, agent, verbose=False, collect_experience=False
    )
    # returns_g2[0] is Minimax's return as P1, returns_g2[1] is Agent's return as P2
    score_minimax_g2 = (
        0.5 if returns_g2[0] == returns_g2[1] else (1.0 if returns_g2[0] > 0 else 0.0)
    )
    score_agent_g2 = 1.0 - score_minimax_g2

    # 更新 Elo (Minimax P1 vs Agent P2)
    new_minimax_p1_elo = update_elo_rating(
        minimax_p1_elo, agent_p2_elo, score_minimax_g2, k_factor
    )
    new_agent_p2_elo = update_elo_rating(
        agent_p2_elo, minimax_p1_elo, score_agent_g2, k_factor
    )

    updated_elos = {
        "agent_p1_elo": new_agent_p1_elo,
        "agent_p2_elo": new_agent_p2_elo,
        "minimax_p1_elo": new_minimax_p1_elo,
        "minimax_p2_elo": new_minimax_p2_elo,
    }

    return updated_elos


def plot_q_values_heatmap(q_values, column_labels=None):
    """
    Plots a heat map of Q-values across columns.

    Parameters:
      q_values (array-like): A 1D array of Q-values (one per column).
      column_labels (list, optional): Labels for each column. If None, numeric labels are used.
    """
    # Convert Q-values to a NumPy array and reshape into a 1xN matrix.
    q_values = np.array(q_values)
    data = q_values.reshape(1, -1)
    # Set up column labels if not provided.
    if column_labels is None:
        column_labels = [str(i) for i in range(data.shape[1])]

    plt.figure(figsize=(8, 2))  # adjust figure size as needed
    # Display the data as an image, 'hot' colormap highlights high values.
    heatmap = plt.imshow(data, cmap="hot", aspect="auto")

    # Add a color bar
    plt.colorbar(heatmap, label="Q-value")

    # Remove the y-axis ticks since there is only one row.
    plt.yticks([])
    # Set x-axis ticks to be at each column center.
    plt.xticks(ticks=np.arange(data.shape[1]), labels=column_labels)

    plt.xlabel("Column")
    plt.title("Heat Map of Q-values Across Columns")
    plt.show()


def plot_visit_counts_heatmap(visits, column_labels=None):
    """
    Plots a heat map of Q-values across columns.

    Parameters:
      q_values (array-like): A 1D array of Q-values (one per column).
      column_labels (list, optional): Labels for each column. If None, numeric labels are used.
    """
    # Convert Q-values to a NumPy array and reshape into a 1xN matrix.
    visits = np.array(visits)
    data = visits.reshape(1, -1)
    # Set up column labels if not provided.
    if column_labels is None:
        column_labels = [str(i) for i in range(data.shape[1])]

    plt.figure(figsize=(8, 2))  # adjust figure size as needed
    # Display the data as an image, 'hot' colormap highlights high values.
    heatmap = plt.imshow(data, cmap="hot", aspect="auto")

    # Add a color bar
    plt.colorbar(heatmap, label="Count")

    # Remove the y-axis ticks since there is only one row.
    plt.yticks([])
    # Set x-axis ticks to be at each column center.
    plt.xticks(ticks=np.arange(data.shape[1]), labels=column_labels)

    plt.xlabel("Column")
    plt.title("Heat Map of Visit Counts Across Columns")
    plt.show()


def progress_to_state(game, move_history):
    """
    Given a game and a move history (list of actions), return the state reached
    by sequentially applying each move from the initial state.
    """
    state = game.new_initial_state()
    for move in move_history:
        if move in state.legal_actions():  # safety check
            state.apply_action(move)
        else:
            raise ValueError(f"Move {move} is not legal in the current state.")
    return state


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
