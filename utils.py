import numpy as np
import pyspiel
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from mcts_wrapper import MCTSWrapper

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
    return obs.reshape(3, 6, 7)

def play_game(agent1, agent2, mcts_wrapper, mcts_simulations, epsilon=0.0, 
              verbose=False, collect_experience=False, replay_buffer=None, 
              lambda_mix=0.5):
    """进行一局游戏，可选择收集经验到回放缓冲区
    
    Args:
        agent1: 玩家1的智能体
        agent2: 玩家2的智能体
        mcts_wrapper: MCTS包装器
        mcts_simulations: MCTS搜索的模拟次数
        epsilon: 探索概率
        verbose: 是否输出游戏过程
        collect_experience: 是否收集经验
        replay_buffer: 经验回放缓冲区
        lambda_mix: MCTS和DQN目标的混合系数
        
    Returns:
        游戏奖励和终局状态
    """
    state = get_initial_state()
    agents = [agent1, agent2]
    
    while not state.is_terminal():
        current_player = state.current_player()
        agent = agents[current_player]
        
        # 获取状态表示
        state_repr = get_state_representation(state, current_player)
        
        # 根据当前状态执行MCTS搜索
        if mcts_simulations > 0:
            mcts_wrapper.max_simulations = mcts_simulations
            best_action, action_q_values, action_counts, _ = mcts_wrapper.search(state)
            
            # 用于探索的概率
            if np.random.random() < epsilon:
                # 随机选择一个动作
                legal_actions = state.legal_actions()
                action = np.random.choice(legal_actions)
            else:
                # 使用MCTS选择的最佳动作
                action = best_action
            
            # 获取MCTS评估的Q值    
            mcts_q_value = action_q_values.get(action, 0.0)
        else:
            # 如果不使用MCTS，直接使用DQN进行动作选择
            action = agent.select_action(state_repr, epsilon)
            mcts_q_value = 0.0  # 没有MCTS评估
        
        # 应用选择的动作
        next_state = state.clone()
        next_state.apply_action(action)
        
        # 计算奖励
        if next_state.is_terminal():
            returns = next_state.returns()
            reward = returns[current_player]
            done = True
        else:
            reward = 0.0
            done = False
        
        # 收集经验到回放缓冲区
        if collect_experience and replay_buffer is not None:
            next_state_repr = get_state_representation(next_state, current_player)
            replay_buffer.add(state_repr, action, reward, next_state_repr, mcts_q_value, done)
        
        # 更新游戏状态
        state = next_state
        
        if verbose:
            print(f"玩家 {current_player} 执行动作 {action}")
            print(state)
    
    # 游戏结束，返回奖励
    returns = state.returns()
    if verbose:
        print(f"游戏结束! 奖励: 玩家0 = {returns[0]}, 玩家1 = {returns[1]}")
    
    return returns, state

def evaluate_agent(agent1, agent2, num_games=100, mcts_simulations=0):
    """评估智能体的性能
    
    Args:
        agent1: 要评估的智能体
        agent2: 对手智能体
        num_games: 评估游戏数量
        mcts_simulations: 评估时使用的MCTS模拟次数
        
    Returns:
        agent1的胜率
    """
    game = get_connect_four_game()
    mcts_wrapper = MCTSWrapper(
        game=game,
        num_simulations=mcts_simulations,
        uct_c=2.0,
        max_nodes=10000,
        dirichlet_alpha=0.0,
        dirichlet_noise=False,
        solve=True
    )
    
    # 玩家1获胜、玩家2获胜和平局的次数
    wins_p1 = 0
    wins_p2 = 0
    draws = 0
    
    for i in range(num_games):
        # 交替先手
        if i % 2 == 0:
            returns, _ = play_game(agent1, agent2, mcts_wrapper, mcts_simulations)
            if returns[0] > 0:
                wins_p1 += 1
            elif returns[1] > 0:
                wins_p2 += 1
            else:
                draws += 1
        else:
            returns, _ = play_game(agent2, agent1, mcts_wrapper, mcts_simulations)
            if returns[0] > 0:
                wins_p2 += 1
            elif returns[1] > 0:
                wins_p1 += 1
            else:
                draws += 1
    
    win_rate = (wins_p1 + 0.5 * draws) / num_games
    return win_rate

def visualize_board(state):
    """可视化Connect4棋盘
    
    Args:
        state: 游戏状态
    """
    # 获取棋盘状态
    board_str = state.observation_string(0)
    rows = board_str.strip().split('\n')
    
    # 转换为数值数组: 0=空格, 1=玩家1(x), 2=玩家2(o)
    board = np.zeros((6, 7), dtype=int)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell == 'x':
                board[i, j] = 1
            elif cell == 'o':
                board[i, j] = 2
    
    # 创建颜色映射: 白色=空, 红色=玩家1, 黄色=玩家2
    cmap = ListedColormap(['white', 'red', 'yellow'])
    
    plt.figure(figsize=(7, 6))
    plt.imshow(board, cmap=cmap)
    plt.grid(color='black', linestyle='-', linewidth=1.5)
    plt.xticks(np.arange(7))
    plt.yticks(np.arange(6))
    plt.title('Connect Four')
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
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid') 