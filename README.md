# Connect4 多智能体强化学习框架

这个项目实现了多种Connect4(四子棋)智能体，包括深度Q网络(DQN)、蒙特卡洛树搜索(MCTS)与DQN结合的混合智能体，以及MiniMax搜索智能体。所有智能体通过统一的接口进行训练和评估。

## 智能体类型

该项目实现了以下智能体类型：

1. **DQNAgent** - 基于深度Q网络的智能体
2. **MCTSDQNAgent** - 结合DQN和MCTS的混合智能体，可选择使用DQN或随机模拟对叶节点进行评估
3. **MiniMaxAgent** - 基于MiniMax算法的智能体，使用Alpha-Beta剪枝优化搜索过程

## 项目架构

项目采用了面向对象的设计模式，主要组件包括：

1. **BaseAgent** - 所有智能体的基类，定义了通用接口
2. **智能体实现** - 继承自BaseAgent的具体智能体实现
3. **经验回放缓冲区** - 存储和采样游戏经验
4. **统一的训练和评估接口** - 使得不同智能体的比较更加公平

## 算法特点

### DQN
- 使用深度卷积网络学习状态-动作价值函数
- 使用经验回放和目标网络提高训练稳定性
- 支持ε-贪婪探索策略

### MCTS-DQN
- 结合MCTS的前瞻搜索能力和DQN的泛化能力
- 可以选择使用DQN评估器或随机模拟评估器
- 混合学习目标提高学习效率

### MiniMax
- 使用经典的MiniMax搜索算法
- Alpha-Beta剪枝减少无效搜索
- 启发式评估函数用于非终端状态评估

## 安装

该项目依赖以下Python库：

- PyTorch
- OpenSpiel
- NumPy
- Matplotlib

可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

项目支持四种主要模式：训练(`train`)、人机对弈(`play`)、AI对战(`agent_vs_agent`)和评估(`evaluate`)。
每种模式都有特定的参数集，可以通过`--help`参数查看详细说明。

### 查看帮助信息

```bash
# 查看主帮助信息
python main.py --help

# 查看特定模式的帮助信息
python main.py train --help
python main.py play --help
python main.py agent_vs_agent --help
python main.py evaluate --help
```

### 训练智能体

```bash
# 训练基本DQN智能体
python main.py train --agent_type dqn --episodes 5000

# 训练MCTS-DQN混合智能体
python main.py train --agent_type mcts_dqn --use_dqn_for_mcts --episodes 10000

# 自定义训练配置
python main.py train --agent_type dqn --episodes 8000 --checkpoint_dir my_models --eval_freq 100 --checkpoint_freq 1000
```

### 人机对弈

```bash
# 与DQN智能体对弈
python main.py play --model_path checkpoints/final_dqn_model.pth --visualize

# 与MCTS增强的DQN智能体对弈
python main.py play --model_path checkpoints/final_dqn_model.pth --mcts_sims 50 --use_dqn_for_mcts --visualize
```

### AI对战

```bash
# DQN vs MiniMax对战
python main.py agent_vs_agent --agent_type dqn --agent_type2 minimax --model_path checkpoints/dqn_model.pth --minimax_depth2 4 --visualize

# DQN vs MCTS+DQN对战
python main.py agent_vs_agent --agent_type dqn --agent_type2 mcts_dqn --model_path checkpoints/dqn_model.pth --model_path2 checkpoints/dqn_model.pth --mcts_sims2 50 --use_dqn_for_mcts2 --visualize

# MiniMax vs MiniMax对战(不同搜索深度)
python main.py agent_vs_agent --agent_type minimax --agent_type2 minimax --minimax_depth 5 --minimax_depth2 3 --visualize
```

### 评估智能体性能

```bash
# 评估MCTS+DQN vs DQN
python main.py evaluate --agent_type mcts_dqn --agent_type2 dqn --model_path checkpoints/dqn_model.pth --model_path2 checkpoints/baseline_model.pth --mcts_sims 50 --use_dqn_for_mcts --num_games 100

# 评估MiniMax vs DQN
python main.py evaluate --agent_type minimax --agent_type2 dqn --model_path2 checkpoints/dqn_model.pth --minimax_depth 4 --num_games 50

# 评估MCTS+DQN vs MiniMax
python main.py evaluate --agent_type mcts_dqn --agent_type2 minimax --model_path 'checkpoints/DQN+MCTS(sim=150)/model_episode_5000.pth' --mcts_sims 50 --use_dqn_for_mcts --minimax_depth2 6 --num_games 10
```

## 文件结构

- `dqn.py` - 包含BaseAgent基类以及所有智能体实现
- `utils.py` - 游戏环境和辅助功能
- `replay_buffer.py` - 经验回放缓冲区
- `mcts_wrapper.py` - MCTS搜索算法的包装器和评估器
- `train.py` - 训练循环
- `main.py` - 主程序入口

## 项目重构

项目进行了全面重构，主要变化包括：

1. 引入BaseAgent基类，实现接口统一
2. 重构DQNAgent，使其继承自BaseAgent
3. 实现MCTSDQNAgent，封装DQN与MCTS的结合
4. 添加MiniMaxAgent作为新的基准智能体
5. 修改训练和评估代码，支持多种智能体类型
6. 简化接口，使用智能体的select_action方法替代复杂的MCTS参数
7. 重构命令行界面，使用子命令模式整理参数，提高用户体验

## 配置参数

各智能体类型支持的配置参数：

### DQN参数
- 学习率、折扣因子、目标网络更新频率等

### MCTS-DQN参数
- 模拟次数、UCT常数、最大节点数
- 评估方式：使用DQN或随机模拟

### MiniMax参数
- 搜索深度
- 评估函数权重

## 性能对比

不同智能体的优势与局限性：

- **DQN**: 学习能力强，但需要大量训练数据；无法进行有效的前瞻搜索
- **MCTS-DQN**: 综合了学习和搜索的优势，性能最佳；但计算开销大，推理速度较慢
- **MiniMax**: 搜索深度有限，但不需要训练；对于中等复杂度的游戏效果良好

## 参考资料

- OpenSpiel文档：[https://openspiel.readthedocs.io](https://openspiel.readthedocs.io)
- DQN论文：[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- AlphaGo Zero：[Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- MiniMax与Alpha-Beta剪枝：[Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/) 