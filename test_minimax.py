import numpy as np
import torch
from dqn import MiniMaxAgent
from utils import get_connect_four_game, get_initial_state, play_game, visualize_board, play_interactive_game

def test_minimax_agent():
    """测试基于OpenSpiel的MiniMaxAgent"""
    print("测试MiniMaxAgent...")
    
    # 创建游戏
    game = get_connect_four_game()
    
    # 创建智能体
    input_shape = (3, 6, 7)  # [通道, 行, 列]
    action_size = 7  # Connect4有7列可以下子
    
    # 创建两个MiniMaxAgent，深度分别为2和4
    agent1 = MiniMaxAgent(input_shape, action_size, max_depth=8)
    agent2 = MiniMaxAgent(input_shape, action_size, max_depth=8)
    
    # 让两个智能体对弈
    print("开始MiniMaxAgentA vs MiniMaxAgentB的对弈...")
    returns, final_state = play_game(agent1, agent2, verbose=True)
    
    # 显示结果
    print("\n最终棋盘状态:")
    print(final_state)
    visualize_board(final_state)
    
    if returns[0] > 0:
        print("MiniMaxAgentA胜利!")
    elif returns[1] > 0:
        print("MiniMaxAgentB胜利!")
    else:
        print("平局!")
    
    # 让人类与AI对弈
    print("\n现在您可以与MiniMaxAgent对弈:")
    agent = MiniMaxAgent(input_shape, action_size, max_depth=4)
    play_interactive_game(agent)

if __name__ == "__main__":
    test_minimax_agent()
