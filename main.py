import argparse
import os
import numpy as np
import torch
import pyspiel

from agents import DQNAgent, MCTSDQNAgent, MiniMaxAgent, BaseAgent
from utils import (
    get_connect_four_game, 
    get_initial_state, 
    play_game, 
    visualize_board,
    play_interactive_game
)
from train import train

def play_against_ai(model_path, mcts_simulations=0, use_dqn_for_mcts=True, visualize=True):
    """让人类玩家与AI对战
    
    Args:
        model_path: DQN模型的路径
        mcts_simulations: MCTS搜索的模拟次数（如果>0，则使用MCTS）
        use_dqn_for_mcts: 是否使用DQN评估MCTS叶节点
        visualize: 是否可视化棋盘
    """
    # 创建游戏
    game = get_connect_four_game()
    
    # 创建智能体
    input_shape = (3, 6, 7)  # [通道, 行, 列]
    action_size = 7  # Connect4有7列可以下子
    
    # 初始化智能体
    if mcts_simulations > 0:
        # 使用MCTSDQNAgent
        agent = MCTSDQNAgent(
            input_shape=input_shape,
            action_size=action_size,
            num_simulations=mcts_simulations,
            use_dqn_evaluator=use_dqn_for_mcts
        )
    else:
        # 使用普通DQNAgent
        agent = DQNAgent(input_shape, action_size)
    
    # 加载模型
    agent.load(model_path)
    
    # 开始交互式游戏
    play_interactive_game(agent)

def play_agent_vs_agent(model_path1, model_path2, mcts_simulations1=0, mcts_simulations2=0, 
                         use_dqn_for_mcts1=True, use_dqn_for_mcts2=True, visualize=True,
                         agent_type1="dqn", agent_type2="dqn", minimax_depth1=4, minimax_depth2=3):
    """让两个AI智能体相互对战
    
    Args:
        model_path1: 第一个DQN模型的路径
        model_path2: 第二个DQN模型的路径
        mcts_simulations1: 第一个智能体的MCTS搜索模拟次数
        mcts_simulations2: 第二个智能体的MCTS搜索模拟次数
        use_dqn_for_mcts1: 第一个智能体是否使用DQN评估MCTS叶节点
        use_dqn_for_mcts2: 第二个智能体是否使用DQN评估MCTS叶节点
        visualize: 是否可视化棋盘
        agent_type1: 第一个智能体的类型
        agent_type2: 第二个智能体的类型
        minimax_depth1: 第一个MiniMax智能体的搜索深度
        minimax_depth2: 第二个MiniMax智能体的搜索深度
    """
    # 创建游戏
    game = get_connect_four_game()
    
    # 创建智能体
    input_shape = (3, 6, 7)  # [通道, 行, 列]
    action_size = 7  # Connect4有7列可以下子
    
    # 初始化两个智能体
    if agent_type1 == "mcts_dqn":
        agent1 = MCTSDQNAgent(
            input_shape=input_shape,
            action_size=action_size,
            num_simulations=mcts_simulations1,
            use_dqn_evaluator=use_dqn_for_mcts1
        )
    elif agent_type1 == "minimax":
        agent1 = MiniMaxAgent(input_shape, action_size, minimax_depth1)
    else:
        agent1 = DQNAgent(input_shape, action_size)
    
    if agent_type2 == "mcts_dqn":
        agent2 = MCTSDQNAgent(
            input_shape=input_shape,
            action_size=action_size,
            num_simulations=mcts_simulations2,
            use_dqn_evaluator=use_dqn_for_mcts2
        )
    elif agent_type2 == "minimax":
        agent2 = MiniMaxAgent(input_shape, action_size, minimax_depth2)
    else:
        agent2 = DQNAgent(input_shape, action_size)
    
    # 加载模型
    if agent_type1 != "minimax":
        agent1.load(model_path1)
    if agent_type2 != "minimax":
        agent2.load(model_path2)
    
    # 开始游戏
    state = get_initial_state()
    print(agent1.max_depth)
    
    while not state.is_terminal():
        # 显示当前棋盘
        print("\n当前棋盘状态:")
        print(state)
        if visualize:
            visualize_board(state)
        
        current_player = state.current_player()
        
        # 获取当前玩家对应的智能体
        current_agent = agent1 if current_player == 0 else agent2
        print(f"智能体{current_player+1}思考中...")
        
        # 获取AI动作
        action = current_agent.select_action(state)
        
        # 检查动作是否合法
        legal_actions = state.legal_actions()
        if action not in legal_actions:
            print(f"警告：智能体 {current_player+1} 选择了非法动作 {action}，改为随机选择合法动作")
            action = np.random.choice(legal_actions)
        
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
    parser.add_argument("--mcts_sims", type=int, default=0, help="MCTS搜索的模拟次数")
    parser.add_argument("--mcts_sims2", type=int, default=0, help="第二个智能体的MCTS搜索模拟次数")
    parser.add_argument("--use_dqn_for_mcts", action="store_true", help="使用DQN评估MCTS叶节点")
    parser.add_argument("--use_dqn_for_mcts2", action="store_true", help="第二个智能体使用DQN评估MCTS叶节点")
    parser.add_argument("--visualize", action="store_true", help="是否可视化棋盘（对弈模式）")
    parser.add_argument("--episodes", type=int, default=5000, help="训练回合数")
    parser.add_argument("--agent_type", choices=["dqn", "mcts_dqn", "minimax"], default="dqn", help="要训练的智能体类型")
    parser.add_argument("--agent_type2", choices=["dqn", "mcts_dqn", "minimax"], default="dqn", help="第二个智能体类型")
    parser.add_argument("--minimax_depth", type=int, default=4, help="MiniMax搜索深度（仅对minimax类型有效）")
    parser.add_argument("--minimax_depth2", type=int, default=3, help="第二个MiniMax智能体的搜索深度")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print(f"开始训练 {args.agent_type} 智能体...")
        
        # 训练配置
        config = {
            "num_episodes": args.episodes,     # 训练回合数
            "batch_size": 64,                  # 训练批次大小
            "replay_buffer_size": 20000,       # 回放缓冲区容量
            "dqn_learning_rate": 0.0001,       # DQN学习率
            "gamma": 0.99,                     # 奖励折扣因子
            "target_update_freq": 100,         # 目标网络更新频率
            "epsilon_start": 1.0,              # 初始探索率
            "epsilon_end": 0.05,               # 最终探索率
            "epsilon_decay": 0.0002,           # 探索率衰减系数
            "training_freq": 4,                # 每回合训练次数
            "lambda_mix": 0.5,                 # MCTS和DQN目标的混合系数
            "eval_freq": 50,                  # 评估频率
            "eval_episodes": 20,               # 评估回合数
            "checkpoint_freq": 500,            # 检查点保存频率
            "checkpoint_dir": "checkpoints",   # 检查点目录
            "device": "cuda" if torch.cuda.is_available() else "cpu",  # 计算设备
            
            # MCTS参数
            "mcts_simulations": 50,            # 每步MCTS模拟次数
            "uct_c": 2.0,                      # UCT探索常数
            "max_nodes": 10000,                # MCTS搜索树最大节点数
            "dirichlet_alpha": 0.3,            # Dirichlet噪声参数
            "dirichlet_noise": True,           # 是否添加Dirichlet噪声
            "solve": True,                     # 是否在MCTS中解决终局状态
            "use_dqn_for_mcts": args.use_dqn_for_mcts,
            "agent_type": args.agent_type      # 智能体类型
        }
        
        # 开始训练
        agent = train(config)
        
    elif args.mode == "play":
        if args.model_path is None or not os.path.exists(args.model_path):
            print("错误: 请提供有效的模型路径")
            return
        
        print(f"使用模型 {args.model_path} 开始游戏...")
        play_against_ai(args.model_path, args.mcts_sims, args.use_dqn_for_mcts, args.visualize)
    
    elif args.mode == "agent_vs_agent":
        # 如果使用minimax智能体，检查model_path是否需要
        if args.agent_type != "minimax" and (args.model_path is None or not os.path.exists(args.model_path)):
            print("错误: 请提供第一个有效的模型路径")
            return
        
        if args.agent_type2 != "minimax" and (args.model_path2 is None or not os.path.exists(args.model_path2)):
            print("错误: 请提供第二个有效的模型路径")
            return
        
        print(f"开始{args.agent_type}智能体 vs {args.agent_type2}智能体的对战...")
        play_agent_vs_agent(
            args.model_path, 
            args.model_path2, 
            args.mcts_sims, 
            args.mcts_sims2,
            args.use_dqn_for_mcts, 
            args.use_dqn_for_mcts2,
            args.visualize,
            args.agent_type,
            args.agent_type2,
            args.minimax_depth,
            args.minimax_depth2
        )

if __name__ == "__main__":
    main() 