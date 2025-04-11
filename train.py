import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

from replay_buffer import ReplayBuffer
from dqn import DQNAgent
from mcts_wrapper import MCTSWrapper
from utils import (
    get_connect_four_game, 
    play_game, 
    evaluate_agent, 
    visualize_board, 
    ensure_dir,
    calculate_moving_average
)

def train(config):
    """训练DQN+MCTS智能体
    
    Args:
        config: 训练配置字典
    """
    # 配置参数
    num_episodes = config["num_episodes"]
    mcts_simulations = config["mcts_simulations"]
    batch_size = config["batch_size"]
    replay_buffer_size = config["replay_buffer_size"]
    dqn_learning_rate = config["dqn_learning_rate"]
    gamma = config["gamma"]
    target_update_freq = config["target_update_freq"]
    epsilon_start = config["epsilon_start"]
    epsilon_end = config["epsilon_end"]
    epsilon_decay = config["epsilon_decay"]
    training_freq = config["training_freq"]
    lambda_mix = config["lambda_mix"]
    eval_freq = config["eval_freq"]
    eval_episodes = config["eval_episodes"]
    checkpoint_freq = config["checkpoint_freq"]
    checkpoint_dir = config["checkpoint_dir"]
    device = config["device"]
    
    # 确保检查点目录存在
    ensure_dir(checkpoint_dir)
    
    # 创建游戏环境
    game = get_connect_four_game()
    
    # 创建MCTS搜索器
    mcts_wrapper = MCTSWrapper(
        game=game,
        num_simulations=mcts_simulations,
        uct_c=config["uct_c"],
        max_nodes=config["max_nodes"],
        dirichlet_alpha=config["dirichlet_alpha"],
        dirichlet_noise=config["dirichlet_noise"],
        solve=config["solve"],
    )
    
    # 创建DQN智能体
    input_shape = (3, 6, 7)  # Connect4的观察空间形状
    action_size = 7  # Connect4有7列可以下子
    
    agent = DQNAgent(
        input_shape=input_shape,
        action_size=action_size,
        device=device,
        learning_rate=dqn_learning_rate,
        gamma=gamma
    )
    
    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
    
    # 记录训练指标
    rewards = []
    losses = []
    win_rates = []
    epsilon_values = []
    
    # 为评估创建一个固定策略的副本
    prev_agent = None
    
    # 训练主循环
    start_time = time.time()
    for episode in range(1, num_episodes + 1):
        # 计算当前的epsilon值（探索率）
        epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
        epsilon_values.append(epsilon)
        
        # 自我对弈并收集经验
        returns, final_state = play_game(
            agent1=agent,
            agent2=agent,
            mcts_wrapper=mcts_wrapper,
            mcts_simulations=mcts_simulations,
            epsilon=epsilon,
            verbose=False,
            collect_experience=True,
            replay_buffer=replay_buffer,
            lambda_mix=lambda_mix
        )
        
        rewards.append(returns[0])  # 记录玩家1的奖励
        
        # 训练DQN智能体
        if len(replay_buffer) >= batch_size:
            for _ in range(training_freq):
                # 从回放缓冲区中采样
                states, actions, rewards_batch, next_states, q_mcts_values, dones = replay_buffer.sample(batch_size)
                
                # 更新DQN
                loss = agent.learn(states, actions, rewards_batch, next_states, q_mcts_values, dones, lambda_mix)
                losses.append(loss)
        
        # 定期更新目标网络
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # 评估智能体性能
        if episode % eval_freq == 0:
            if prev_agent is None:
                # 首次评估，与自己对弈
                win_rate = 0.5  # 理论上与自己对弈应为50%胜率
            else:
                # 与之前的智能体对弈
                win_rate = evaluate_agent(agent, prev_agent, num_games=eval_episodes, mcts_simulations=0)
            
            win_rates.append(win_rate)
            
            # 打印训练进度
            elapsed_time = time.time() - start_time
            avg_loss = np.mean(losses[-training_freq*batch_size:]) if losses else 0
            print(f"回合 {episode}/{num_episodes} | 时间: {elapsed_time:.1f}s | 探索率: {epsilon:.4f} | 胜率: {win_rate:.4f} | 损失: {avg_loss:.6f}")
            
            # 创建一个新的评估基准
            if episode % (eval_freq * 5) == 0:
                prev_agent = deepcopy(agent)
        
        # 保存检查点
        if episode % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_mcts_episode_{episode}.pt")
            agent.save(checkpoint_path)
    
    # 训练结束，保存最终模型
    final_model_path = os.path.join(checkpoint_dir, "dqn_mcts_final.pt")
    agent.save(final_model_path)
    
    # 绘制训练指标
    if losses:
        plt.figure(figsize=(15, 10))
        
        # 绘制损失曲线
        plt.subplot(3, 1, 1)
        plt.plot(calculate_moving_average(losses, window=100))
        plt.title('Training Loss (Moving Average)')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        
        # 绘制探索率曲线
        plt.subplot(3, 1, 2)
        plt.plot(epsilon_values)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        
        # 绘制胜率曲线
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(eval_freq, num_episodes + 1, eval_freq), win_rates)
        plt.title('Win Rate')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, 'training_metrics.png'))
        plt.show()
    
    return agent

if __name__ == "__main__":
    # 训练配置
    config = {
        "num_episodes": 20000,          # 训练回合数(从5000增加到20000)
        "mcts_simulations": 500,        # 每步MCTS模拟次数(从50增加到500)
        "batch_size": 64,               # 训练批次大小
        "replay_buffer_size": 50000,    # 回放缓冲区容量(从20000增加到50000)
        "dqn_learning_rate": 0.0001,    # DQN学习率
        "gamma": 0.99,                  # 奖励折扣因子
        "target_update_freq": 100,      # 目标网络更新频率
        "epsilon_start": 1.0,           # 初始探索率
        "epsilon_end": 0.1,             # 最终探索率(从0.05增加到0.1)
        "epsilon_decay": 0.0001,        # 探索率衰减系数(从0.0002减小到0.0001)
        "training_freq": 4,             # 每回合训练次数
        "lambda_mix": 0.7,              # MCTS和DQN目标的混合系数(从0.5增加到0.7)
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