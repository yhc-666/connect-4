import argparse
import os
import numpy as np
import torch
import pyspiel

from dqn import DQNAgent
from mcts_wrapper import MCTSWrapper
from utils import (
    get_connect_four_game, 
    get_initial_state, 
    play_game, 
    visualize_board
)
from train import train

def play_against_ai(model_path, mcts_simulations=0, visualize=True):
    """让人类玩家与AI对战
    
    Args:
        model_path: DQN模型的路径
        mcts_simulations: MCTS搜索的模拟次数
        visualize: 是否可视化棋盘
    """
    # 创建游戏
    game = get_connect_four_game()
    
    # 创建智能体
    input_shape = (3, 6, 7)  # [通道, 行, 列]
    action_size = 7  # Connect4有7列可以下子
    
    # 初始化智能体并加载模型
    agent = DQNAgent(input_shape, action_size)
    agent.load(model_path)
    
    # 创建MCTS搜索器
    mcts_wrapper = MCTSWrapper(
        game=game,
        num_simulations=mcts_simulations,
        uct_c=2.0,
        max_nodes=10000,
        dirichlet_alpha=0.0,
        dirichlet_noise=False,
        solve=True
    )
    
    # 开始游戏
    state = get_initial_state()
    
    while not state.is_terminal():
        # 显示当前棋盘
        print("\n当前棋盘状态:")
        print(state)
        
        current_player = state.current_player()
        
        if current_player == 0:  # 人类玩家
            # 获取合法动作
            legal_actions = state.legal_actions()
            
            # 如果没有合法动作，跳过
            if not legal_actions:
                print("没有合法动作，游戏结束!")
                break
            
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
        else:  # AI玩家
            print("AI思考中...")
            
            # 获取AI动作
            if mcts_simulations > 0:
                # 使用MCTS搜索
                best_action, _, action_counts, _ = mcts_wrapper.search(state)
                action = best_action
                
                # 打印搜索统计信息
                print(f"MCTS访问次数: {action_counts}")
            else:
                # 仅使用DQN策略
                state_repr = mcts_wrapper.get_state_representation(state, current_player)
                action = agent.select_action(state_repr)
        
        # 应用选中的动作
        print(f"玩家 {current_player} 选择列 {action}")
        state.apply_action(action)
    
    # 游戏结束
    print("\n最终棋盘状态:")
    print(state)
    if visualize:
        visualize_board(state)
    
    # 显示结果
    returns = state.returns()
    if returns[0] > 0:
        print("你赢了!")
    elif returns[1] > 0:
        print("AI赢了!")
    else:
        print("平局!")

def play_agent_vs_agent(model_path1, model_path2, mcts_simulations1=0, mcts_simulations2=0, visualize=True):
    """让两个AI智能体相互对战
    
    Args:
        model_path1: 第一个DQN模型的路径
        model_path2: 第二个DQN模型的路径
        mcts_simulations1: 第一个智能体的MCTS搜索模拟次数
        mcts_simulations2: 第二个智能体的MCTS搜索模拟次数
        visualize: 是否可视化棋盘
    """
    # 创建游戏
    game = get_connect_four_game()
    
    # 创建智能体
    input_shape = (3, 6, 7)  # [通道, 行, 列]
    action_size = 7  # Connect4有7列可以下子
    
    # 初始化两个智能体并加载模型
    agent1 = DQNAgent(input_shape, action_size)
    agent1.load(model_path1)
    
    agent2 = DQNAgent(input_shape, action_size)
    agent2.load(model_path2)
    
    # 创建两个MCTS搜索器
    mcts_wrapper1 = MCTSWrapper(
        game=game,
        num_simulations=mcts_simulations1,
        uct_c=2.0,
        max_nodes=10000,
        dirichlet_alpha=0.0,
        dirichlet_noise=False,
        solve=True
    )
    
    mcts_wrapper2 = MCTSWrapper(
        game=game,
        num_simulations=mcts_simulations2,
        uct_c=2.0,
        max_nodes=10000,
        dirichlet_alpha=0.0,
        dirichlet_noise=False,
        solve=True
    )
    
    # 开始游戏
    state = get_initial_state()
    
    while not state.is_terminal():
        # 显示当前棋盘
        print("\n当前棋盘状态:")
        print(state)
        if visualize:
            visualize_board(state)
        
        current_player = state.current_player()
        
        # 获取当前玩家对应的智能体和MCTS搜索器
        if current_player == 0:
            current_agent = agent1
            current_mcts = mcts_wrapper1
            current_sims = mcts_simulations1
            print("智能体1思考中...")
        else:
            current_agent = agent2
            current_mcts = mcts_wrapper2
            current_sims = mcts_simulations2
            print("智能体2思考中...")
        
        # 获取AI动作
        if (current_player == 0 and mcts_simulations1 > 0) or (current_player == 1 and mcts_simulations2 > 0):
            # 使用MCTS搜索
            best_action, _, action_counts, _ = current_mcts.search(state)
            action = best_action
            
            # 打印搜索统计信息
            print(f"MCTS访问次数: {action_counts}")
        else:
            # 仅使用DQN策略
            state_repr = current_mcts.get_state_representation(state, current_player)
            action = current_agent.select_action(state_repr)
        
        # 应用选中的动作
        print(f"智能体 {current_player+1} 选择列 {action}")
        state.apply_action(action)
    
    # 游戏结束
    print("\n最终棋盘状态:")
    print(state)
    if visualize:
        visualize_board(state)
    
    # 显示结果
    returns = state.returns()
    if returns[0] > 0:
        print("智能体1赢了!")
    elif returns[1] > 0:
        print("智能体2赢了!")
    else:
        print("平局!")

def main():
    parser = argparse.ArgumentParser(description="DQN+MCTS Connect4")
    parser.add_argument("--mode", choices=["train", "play", "agent_vs_agent"], default="train", help="运行模式: 训练或对弈")
    parser.add_argument("--model_path", type=str, default=None, help="要加载的模型路径（对弈模式）")
    parser.add_argument("--model_path2", type=str, default=None, help="第二个模型的路径（agent_vs_agent模式）")
    parser.add_argument("--mcts_sims", type=int, default=1000, help="MCTS搜索的模拟次数")
    parser.add_argument("--mcts_sims2", type=int, default=1000, help="第二个智能体的MCTS搜索模拟次数")
    parser.add_argument("--visualize", action="store_true", help="是否可视化棋盘（对弈模式）")
    args = parser.parse_args()
    
    if args.mode == "train":
        print("开始训练DQN+MCTS智能体...")
        
        # 训练配置
        config = {
            "num_episodes": 5000,           # 训练回合数
            "mcts_simulations": 50,         # 每步MCTS模拟次数
            "batch_size": 64,               # 训练批次大小
            "replay_buffer_size": 20000,    # 回放缓冲区容量
            "dqn_learning_rate": 0.0001,    # DQN学习率
            "gamma": 0.99,                  # 奖励折扣因子
            "target_update_freq": 100,      # 目标网络更新频率
            "epsilon_start": 1.0,           # 初始探索率
            "epsilon_end": 0.05,            # 最终探索率
            "epsilon_decay": 0.0002,        # 探索率衰减系数
            "training_freq": 4,             # 每回合训练次数
            "lambda_mix": 0.5,              # MCTS和DQN目标的混合系数
            "eval_freq": 200,               # 评估频率
            "eval_episodes": 20,            # 评估回合数
            "checkpoint_freq": 500,         # 检查点保存频率
            "checkpoint_dir": "checkpoints",# 检查点目录
            "device": "cuda" if torch.cuda.is_available() else "cpu",  # 计算设备
            
            # MCTS参数
            "uct_c": 2.0,                   # UCT探索常数
            "max_nodes": 10000,             # MCTS搜索树最大节点数
            "dirichlet_alpha": 0.3,         # Dirichlet噪声参数
            "dirichlet_noise": True,        # 是否添加Dirichlet噪声
            "solve": True,                  # 是否在MCTS中解决终局状态
        }
        
        # 开始训练
        agent = train(config)
        
    elif args.mode == "play":
        if args.model_path is None or not os.path.exists(args.model_path):
            print("错误: 请提供有效的模型路径")
            return
        
        print(f"使用模型 {args.model_path} 开始游戏...")
        play_against_ai(args.model_path, args.mcts_sims, args.visualize)
    
    elif args.mode == "agent_vs_agent":
        if args.model_path is None or not os.path.exists(args.model_path):
            print("错误: 请提供第一个有效的模型路径")
            return
        
        if args.model_path2 is None or not os.path.exists(args.model_path2):
            print("错误: 请提供第二个有效的模型路径")
            return
        
        print(f"使用模型 {args.model_path} 和 {args.model_path2} 开始AI对战...")
        play_agent_vs_agent(args.model_path, args.model_path2, args.mcts_sims, args.mcts_sims2, args.visualize)

if __name__ == "__main__":
    main() 