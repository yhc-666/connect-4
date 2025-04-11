# Connect4 DQN+MCTS 自我对弈强化学习

这个项目实现了一个基于深度Q网络(DQN)和蒙特卡洛树搜索(MCTS)的Connect4(四子棋)智能体，通过自我对弈的方式学习游戏策略。

## 项目组件

该项目包含以下主要组件：

1. **DQN模型** - 用于学习状态-动作价值函数
2. **MCTS搜索** - 提供前瞻搜索能力
3. **经验回放缓冲区** - 存储和采样游戏经验
4. **自我对弈训练循环** - 协调训练过程

## 算法特点

本项目的核心是将MCTS与DQN相结合，利用两者的优势：

- **MCTS** 提供高质量的搜索结果，可以看到几步之后的游戏局面
- **DQN** 提供一般化的价值函数，能够从经验中学习
- **混合学习目标** 将MCTS的搜索值与DQN的时序差分(TD)目标相结合，提高学习效率

学习目标公式：
```
y = (1 - λ)(r + γ * max(Q(s',a'))) + λ * Q_MCTS(s,a)
```

## 安装

该项目依赖以下Python库：

- PyTorch
- OpenSpiel
- NumPy
- Matplotlib

可以通过以下命令安装依赖：

```bash
pip install torch numpy matplotlib
pip install open_spiel
```

## 使用方法

### 训练模型

要启动训练过程，请运行：

```bash
python main.py --mode train
```

训练参数可以在`main.py`中的配置字典中修改。

### AI与AI对弈
```bash
python main.py --mode agent_vs_agent --model_path checkpoints/dqn_mcts_final.pt --model_path2 checkpoints/dqn_mcts_final.pt --mcts_sims 50 --mcts_sims2 50 --visualize
```

### 与AI对弈

训练完成后，可以通过以下命令与训练好的AI对弈：

```bash
python main.py --mode play --model_path checkpoints/dqn_mcts_final.pt --visualize
```

参数说明：
- `--model_path`: 训练好的模型路径
- `--mcts_sims`: 对弈时MCTS的模拟次数（默认50），0表示仅使用DQN
- `--visualize`: 是否显示棋盘可视化

## 文件结构

- `dqn.py` - DQN网络模型和智能体
- `mcts_wrapper.py` - MCTS搜索算法的包装器
- `replay_buffer.py` - 经验回放缓冲区
- `utils.py` - 游戏环境和辅助功能
- `train.py` - 训练循环
- `main.py` - 主程序入口

## 配置参数

训练配置参数包括：

- **DQN参数**：学习率、折扣因子、目标网络更新频率等
- **MCTS参数**：模拟次数、UCT常数、最大节点数等
- **训练参数**：回合数、批次大小、探索率等

## 训练过程

训练过程会记录以下指标：

1. **损失曲线** - 网络训练损失
2. **探索率** - Epsilon随时间的变化
3. **胜率** - 对战历史版本的胜率

模型会定期保存到`checkpoints`目录。

## 性能指标

模型的性能可以通过以下方式评估：

- 与随机策略对弈的胜率
- 与旧版本的模型对弈的胜率
- 在游戏树的搜索深度和准确性

## 参考资料

- OpenSpiel文档：[https://openspiel.readthedocs.io](https://openspiel.readthedocs.io)
- DQN论文：[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- AlphaGo Zero：[Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) 