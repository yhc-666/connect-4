# 

## Policy based (PPO)+MCTS

通常的实践中，MCTS和PPO结合时：

- PPO负责生成一个初始策略（Actor）和初始价值函数（Critic）（PPO中的critic函数是用于生成优势函数中的baseline）。
- MCTS使用PPO提供的策略和价值函数进行搜索和规划：
  - 改进价值估计；
  - 产生更好的策略（动作分布）。
- 然后MCTS产生的数据（或策略）可用于：
  - 政策蒸馏（Policy Distillation）监督PPO的策略网络更新；
  - 辅助训练Critic网络。

而并非简单直接替换PPO的MC回报。

> ## 具体区别体现在哪里？
>
> | 方面         | 传统PPO（MC Return监督） | PPO+MCTS（策略蒸馏监督） |
> | ------------ | ------------------------ | ------------------------ |
> | 学习类型     | 强化学习 (RL)            | 监督学习 (Imitation)     |
> | 信号来源     | 环境真实回报 (MC Return) | MCTS模拟生成的策略分布   |
> | 优化目标     | 直接最大化累计奖励       | 模仿MCTS给定策略         |
> | 探索能力     | 强（直接环境互动）       | 较弱（取决于MCTS质量）   |
> | 策略质量依赖 | 环境回报准确性           | MCTS搜索能力与质量       |

> ## PPO中加入MCTS策略蒸馏的两种实现方式（混合模式）：
>
> 你完全可以采用**混合**方式：
>
> - **Critic网络**仍然使用**MC回报或MCTS模拟回报**训练；
> - **Actor网络**则同时考虑两个损失：
>   - PPO原始策略梯度损失（使用Advantage）；
>   - 策略蒸馏损失（监督学习模仿MCTS策略）；
>
> 例如，最终Actor网络损失可能是：
>
> $L_{actor}^{total}(\theta) = \alpha \cdot L_{actor}^{PPO}(\theta) + (1-\alpha)\cdot L_{distillation}(\theta)$
>
> 其中：
>
> - $L_{actor}^{PPO}(\theta)$ 是传统PPO策略梯度损失；
> - $L_{distillation}(\theta)$是模仿MCTS策略的交叉熵损失；
> - α是权重，控制强化学习和监督学习之间的比例。



是的，DQN相比Policy-based方法更适合与MCTS结合。
原因在于：

- **DQN提供明确的价值估计（Q值）**，天然匹配MCTS中对节点状态价值的评估需求，而Policy-based方法主要输出策略概率，价值估计相对模糊。
- MCTS的核心优势在于通过搜索精确地修正动作价值，而DQN本身的价值导向特性与这种修正方式更直接兼容。

## Value based (DQN)+MCTS









## Solved

**7列×6行**的标准Connect4棋盘下：

- 如果先手在**中央列**开始游戏，**且每一步都使用完美策略**，先手一定能获胜。
- 如果先手在其他列开始，则对手可能至少逼平，甚至翻盘。



compare alpha zero vs DQN + MCTS





## MCTS + DQN selfplay

### 思路

我们要在**Connect4**环境下让智能体进行自我对弈，每一步动作由**MCTS**辅助决策，同时通过**DQN**进行价值函数学习，利用对弈产生的数据不断更新网络。在此过程中，**MCTS**与**DQN**相互配合：

1. **MCTS**：
   - 在每一步决策时，使用当前DQN作为叶节点的评估函数或辅助Rollout，进行若干次模拟，选出“更优”的动作。
   - 将该动作、状态转移以及搜索过程中得到的估值信息提供给DQN来优化TD target。
2. **DQN**：
   - 作为价值函数近似器，对状态-动作对进行Q估计。
   - 从Replay Buffer中采样训练数据，更新参数，使得网络的预测Q与（MCTS搜索到的）真实价值或混合目标尽可能一致。

通过不断**自我对弈**，我们可以让智能体在没有外部对手（或专家演示）的情况下，逐渐掌握策略。

### 算法流程

以下以 **单次对局**（从空棋盘到终局）为单位，描述完整流程。

1. **初始化**：
   - 建立 DQN 网络 $$Q_\theta(s,a)$$ 并初始化参数 $$\theta$$。  
   - 准备 Replay Buffer $$\mathcal{D}$$。  
   - 设置超参数：折扣因子 $$\gamma$$、学习率 $$\alpha$$、MCTS 模拟次数 $$N_{\text{sim}}$$、混合系数 $$\lambda$$ 等。

2. **自我对弈 (Self-play)**：
   - 从空棋盘开始，轮到当前玩家执行动作时：
     1. **MCTS**：以当前状态为根节点，展开 MCTS 搜索 $$N_{\text{sim}}$$ 次。  
        - 在 MCTS 叶节点利用 **DQN** 估计该叶子状态的价值（或进行有限步随机 Rollout）。  
        
          > **Early Training (Rollout)**
          >
          > - Use **short, cheap rollouts** at MCTS leaf nodes.
          > - This “bootstraps” the training, because your DQN is initially untrained or very inaccurate. Rollouts help provide somewhat more stable (though often noisy) value estimates.
          > - As you gather more self-play data, the network parameters θ\thetaθ start to converge to better Q-values.
          >
          > **Later Training (Q-value)**
          >
          > - Gradually **reduce or disable rollouts** in favor of the **DQN’s Q-value** estimate.
          > - Once the DQN is reasonably accurate, relying on its value predictions is typically more **computationally efficient** (no extra simulation steps per leaf), and it ensures that the agent is leveraging its learned knowledge.
        
        - 回溯更新得到根节点各动作的访问次数与价值。  
        
     2. 在根节点上，**基于访问次数或价值** 选择最终动作 $$a$$。  
     
     3. 环境执行动作 $$a$$，获得下一个状态 $$s'$$ 及即时奖励 $$r$$（如胜利 +1、失败 -1、平局 0 等）。  
     
     4. **存储数据**：将 $$\bigl(s, a, r, s'\bigr)$$ 及根节点的 $$Q_{\text{MCTS}}(s,a)$$存入 Replay Buffer $$\mathcal{D}$$。  
     
   - 如果达到终局（棋盘满或有人获胜），结束该对局。
   
3. **网络训练 (DQN Update)**：
   
   - 在对局或数步后，定期从 Replay Buffer 中随机采样 (Batch) 若干条 $$\bigl(s, a, r, s'\bigr)$$。  
   - **计算训练目标**：  
     - **基础 DQN 目标**：  
       $$
       y_{\text{DQN}} = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
       $$
       其中 $$\theta^-$$ 是目标网络的参数，用于稳定训练。  
     - **MCTS 搜索估值**：  
       $$
       Q_{\text{MCTS}}(s,a)
       $$
       为在根节点对动作 $$a$$ 的搜索 $$Q$$ 值。  
     - **混合目标**（为减少 MCTS 估值波动的影响）：  
       $$
       y = (1 - \lambda) \cdot y_{\text{DQN}} \;\;+\;\; \lambda \cdot Q_{\text{MCTS}}(s,a)
       $$
       其中 $$\lambda \in [0,1]$$。  
   - **梯度下降**：令  
     $$
     \mathcal{L}(\theta) \;=\; \frac{1}{|\text{Batch}|} \sum_{(s,a,r,s')\in \text{Batch}} \Bigl(Q_{\theta}(s,a) - y\Bigr)^2
     $$
   - 定期将 $$\theta$$ 同步给 $$\theta^-$$（目标网络）。
   
4. **重复以上过程**：  
   - 经过多轮自我对弈，智能体不断积累并学习新的数据，**DQN** 与 **MCTS** 互相促进，策略逐渐收敛。





1. 



### 设计理由

针对 **DQN + MCTS** 自我对弈在 Connect4 问题上的设计，以下是主要的理由与注意事项，分点列出以便查阅：

1. **融合 $$Q_{\text{MCTS}}$$ 与 DQN 的目标**
   
   - **问题背景**  
     
   - 如果**只**用DQN的target network没法利用MCTS的refine
     
   - 如果**只**将 MCTS 根节点的 $$Q_{\text{MCTS}}(s,a)$$ 作为网络的唯一监督信号，容易因搜索不充分而带来高偏差，尤其在训练初期或当状态空间较大时。而且**MCTS 搜索估值可能不稳定**，，MCTS 在局部状态下可能高估或低估某些动作的价值。
     
     - **原因**  
       受限于模拟次数、搜索深度和环境复杂度，MCTS 在局部状态下可能高估或低估某些动作的价值。
     - 导致训练不稳定难以收敛/效率不好
     
   - **解决方案**  
     使用混合目标  
     $$
     y \;=\; (1 - \lambda)\cdot \Bigl(r \;+\;\gamma \max_{a'}Q_{\theta^-}(s',a')\Bigr) \;+\;\lambda \cdot Q_{\text{MCTS}}(s,a),
     $$
     其中 $$\lambda \in [0,1]$$ 是控制搜索引导和传统 TD 目标平衡的关键超参数。
   
1. 在MCTS算法中估算MCTS 叶节点时训练前期采用rollout后期采用DQN，因为.....
   
3. **目标网络 (Target Network) 的必要性**
   - **DQN 稳定机制**  
     在计算 $$r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$ 时使用旧的网络参数 $$\theta^-$$，减少自举偏差的累积。  
   - **实现**  
     定期将在线网络的参数 $$\theta$$ 复制给目标网络 $$\theta^-$$（如每隔固定轮数或固定时间），以在训练中维持稳定性。

4. **自我对弈 (Self-play) 策略**
   - **原因**  
     - 若无固定对手或外部示例，自我对弈是常见的博弈强化学习手段(inspired byu alphazero)；  
     - 智能体对手随自身水平提高而提高，能持续提供有挑战性的对局。
   - **实现**  
     - 双方均使用当前或同一版本的网络 + MCTS 搜索决策；  
     - 每一步均将状态、动作、奖励、MCTS 估值等数据存入 Replay Buffer，完成后进行 DQN 训练。

6. **整体收益**
   - **优势互补**  
     - MCTS 提供更准确的搜索决策；  
     - DQN 提供对大规模状态空间的泛化能力。  
   - **训练过程**  
     自我对弈将不断提升策略水平，MCTS 与 DQN 互相促进，帮助网络学到更接近最优策略的价值函数。

通过以上分点说明，可以更清晰地理解各项设计背后的**动机**与**执行细节**。在实际实现中，根据具体实验需求和资源限制，对这些要点进行适度微调即可。 







**资源与超参数取舍**

- **MCTS 模拟次数**：  
  $$N_{\text{sim}}$$ 越多，搜索越精确，但计算量也更大。  
- **混合系数** $$\lambda$$：  
  需根据实验在 $$[0,1]$$ 区间内调参，视搜素可靠度和网络稳定性而定。  
- **训练频率、批大小、学习率**：  
  这些常规 DQN 超参数也决定了收敛速度和稳定性，需要结合搜索所消耗的时间进行整体权衡。

DQN play first: winrate=0.28, drawrate=0.00, loserate=0.72
DQN play second: winrate=0.34, drawrate=0.00, loserate=0.66



与纯MCTS/DQN对比







错误原因：当游戏到达终止状态时，current_player() 返回 kTerminalPlayerId（-4），而尝试用这个负值作为索引去访问rewards()返回的列表，导致索引越界。









## Optimal strategy

**Game Solved**: On a standard 7×6 board, the first player can force a win with optimal moves.

**Central Priority**: Start in the middle columns (ideally the central column) to maximize potential four-in-a-row opportunities.

**Double Threats**: Aim to create situations where you threaten to complete four in a row in two different places, making it impossible for the opponent to defend both with one move.

**Forced Moves**: React promptly to opponent threats while methodically building your own; sacrifice immediate gains if it means setting up a winning fork in the next turn.

**Consistent Pressure**: Maintain control by continuously generating new threats, especially around the center columns, ensuring the opponent is always on the defensive.





## Method







## Experiment

我决定在report的实验部分这样写包含以下：

对局演示：against human player

**1.Training Analysis**

Let each agent (e.g., “DQN” and “MiniMax”) alternate between going first and second. You then collect all the results (wins/losses/draws) into a single sequence to compute a single Elo rating per agent.

1.1Elo曲线图：

**定期评估**

- 每隔指定数量的 episodes（如 `evaluation_interval = 1000`），暂停训练，进行一次评估。
- 评估时，让当前的 Agent 与 Minimax 先手后手各对战一局（如 `num_evaluation_games = 50`），记录对战结果（胜=1, 平=0.5, 负=0）。

**Elo 计算**

- 为 Agent 和 Minimax 都维护一个 Elo 分数（先手/后手分别），初始值一般设为 1200。
- 每场评估对战完成后，使用 Elo 公式（见下文）更新双方 Elo。

- X轴（横坐标）：

  

  - 智能体的训练episodes或训练局数。

    

- Y轴（纵坐标）：

  

  - 智能体和Minimax算法的实时Elo值。

    

- 图像含义：

  

  - Agent Elo高于Minimax Elo → 你的Agent实力已超过Minimax。

    

  - Elo曲线上升趋势 → 智能体性能持续提高。

1.2Loss 曲线图

证明训练有效且收敛

2.**Comparison with Baselines**

|                            | Playing as first | Playing as Second |       |      |       |       |
| -------------------------- | ---------------- | ----------------- | ----- | ---- | ----- | ----- |
| Agent                      | Win%             | Draw%             | Loss% | Win% | Draw% | Loss% |
| DQN + MCTS (play w/ MCTS)  |                  |                   |       |      |       |       |
| DQN + MCTS (play w/o MCTS) |                  |                   |       |      |       |       |
| DQN                        |                  |                   |       |      |       |       |
| Pure MCTS                  |                  |                   |       |      |       |       |

Table: Evaluate agent against MiniMax agent, each pair repeat for 100 episodes

**3.Abalation Studies**

lambda = 0

lambda = 0.5

lambda = 1

lambda权重调为0，0.5，1 看看只使用MCTS Q作为target，加权，只使用Target network Q的效果不同，证明我提出的Q融合有效

Simulation = 10

Simulation = 50

Simulation = 150

证明MCTS有帮助

4.Heat Map,证明agent学到了optimal策略比如，先手选中间；对手要赢的时候堵人









### 1. Training Analysis

1. **Elo 曲线**

   - **目标**：衡量智能体随训练进程的实力变化，并与depth=6 Minimax 算法做对比。

   - **方法**：

     1. 每完成50个 Episode（或对局）后，与固定难度的 depth=6 Minimax 对手先后手各对战一局

     2. 将智能体的胜率或对战结果转化为 Elo 评分；Minimax 作为对照组，也可计算其 Elo 值或用固定基准 Elo。

     3. 分为先后手分别记录 graph1 (x: episodes y: elo of agent play first minimax play second)

        graph2 (x: episodes y: elo of agent play second minimax play first)

   - **结果预期**：

     - 如果智能体实力超过 Minimax，则在某个训练阶段后，Elo 曲线会超过 Minimax 参照线。
     - 智能体的 Elo 随着训练推进逐渐上升，说明策略在不断改进。

2. **Loss 曲线**

   - **目标**：观察神经网络在训练过程中的收敛情况。
   - **方法**：
     1. 记录每个训练迭代（或每隔一定 Batch）的平均损失值（Loss）。
     2. 以迭代轮数或Episode 数为横坐标，损失值为纵坐标，绘制曲线图。
   - **结果预期**：
     - Loss 随训练进行而逐渐下降，波动幅度可接受，说明模型在学习有效策略并趋于稳定。

------

### 2. Comparison with Baseline: Minimax with Depth=7

|                                      | Playing as First | Playing as Second |       |      |       |       |
| ------------------------------------ | :--------------- | :---------------- | ----- | ---- | ----- | ----- |
| **Agent**                            | Win%             | Draw%             | Loss% | Win% | Draw% | Loss% |
| **DQN + MCTS**  (inference w/ MCTS)  |                  |                   |       |      |       |       |
| **DQN + MCTS**  (inference w/o MCTS) |                  |                   |       |      |       |       |
| Pure **DQN**                         |                  |                   |       |      |       |       |
| **Pure MCTS**                        |                  |                   |       |      |       |       |

------

### 3. Ablation Study

1. **不同 λ\lambdaλ（混合系数）**

   - **设置**：

     λ∈{ 0,  0.5,  1}  \lambda \in \{\, 0, \; 0.5, \; 1 \}λ∈{0,0.5,1}

     - λ=0\lambda = 0λ=0：仅用传统 DQN 目标；
     - λ=0.5\lambda = 0.5λ=0.5：DQN 目标与 MCTS 估值各占 50%；
     - λ=1\lambda = 1λ=1：完全使用 MCTS 给出的 Q 值为训练目标。

   - **结果指标**：在相同环境设置下，对比训练时的loss收敛曲线与和baseline（d=7 minimax）的对战结果。

     - Graph ：x：episodes y：loss with λ=0/0.5/1
     - Table: 

   - **结果解读**：

     - 若 λ=0.5\lambda = 0.5λ=0.5 表现最佳，说明结合两种 Q 值估计能更有效地训练网络。
     - λ=0\lambda = 0λ=0 和 λ=1\lambda = 1λ=1 的对比能体现只依赖单一价值来源时的优劣。

2. **不同 MCTS 模拟次数**

   - **设置**：{ 10,  50,  150}\{\,10, \; 50, \; 150\}{10,50,150} 次模拟
   - **结果指标**：在相同 λ\lambdaλ 下，对比不同模拟次数对应的胜率/Elo、搜索时长，以及 Loss 收敛速度。
   - **结果解读**：
     - 模拟次数太低，可能导致搜索深度不足；太高则增加计算量，但若带来显著胜率提升，也说明 MCTS 对性能贡献大。
     - 综合考虑性能与计算成本，找到合适的模拟次数范围。

------

### 4. 实验小结

通过以上实验，初步结论如下：

1. **训练有效性**：
   - Elo 曲线与 Loss 曲线证明智能体整体实力随训练迭代持续增强，并且收敛稳定。
2. **与 Baseline 的比较**：
   - DQN + MCTS 结合在多数情况下优于纯 MCTS 或纯 DQN，胜率或 Elo 也超过 Minimax 对手。
   - 在同样的搜索或运算预算下，DQN + MCTS 能更好地学习对策并做出深度搜索决策。
3. **消融实验**：
   - λ\lambdaλ 不同取值显示：同时利用 DQN 与 MCTS 提供的估值可提升策略质量；过度依赖单方可能导致稳定性或精度受限。
   - MCTS 模拟次数会显著影响搜索深度与决策质量；适度增加模拟次数往往能提升胜率，但也提高了计算开销，需要在实际部署中平衡。





