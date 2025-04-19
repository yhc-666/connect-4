import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BaseAgent:
    """所有智能体的基类"""

    def __init__(self, input_shape, action_size, device="cpu"):
        """基础智能体初始化

        Args:
            input_shape: 输入状态的形状
            action_size: 动作空间大小
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    def select_action(self, state, epsilon=0.0):
        """选择动作

        Args:
            state: 当前状态
            epsilon: 探索概率

        Returns:
            选择的动作
        """
        raise NotImplementedError("子类必须实现select_action方法")

    def save(self, path):
        """保存模型参数"""
        raise NotImplementedError("子类必须实现save方法")

    def load(self, path):
        """加载模型参数"""
        raise NotImplementedError("子类必须实现load方法")


class DQN(nn.Module):
    """深度Q网络模型"""

    def __init__(self, input_shape, action_size):
        """初始化DQN网络

        Args:
            input_shape: 输入状态的形状（行，列，通道）
            action_size: 动作空间大小
        """
        super(DQN, self).__init__()

        # Connect4游戏状态的特征提取器
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 计算卷积层输出尺寸
        conv_output_size = 128 * input_shape[1] * input_shape[2]

        # Q值预测头
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入状态表示，形状为 (batch_size, channels, rows, cols)

        Returns:
            每个动作的Q值预测
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent(BaseAgent):
    """DQN智能体"""

    def __init__(
        self,
        input_shape,
        action_size,
        device="cpu",
        learning_rate=0.001,
        gamma=0.99,
        tau=0.001,
        n_step=1,
    ):
        """初始化DQN智能体

        Args:
            input_shape: 输入状态的形状
            action_size: 动作空间大小
            device: 计算设备 ('cpu' 或 'cuda')
            learning_rate: 学习率
            gamma: 奖励折扣因子
            tau: 目标网络软更新系数
        """
        super(DQNAgent, self).__init__(input_shape, action_size, device)
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 目标网络软更新系数
        self.n_step = n_step

        # 创建主网络和目标网络
        self.policy_net = DQN(input_shape, action_size).to(self.device)
        self.target_net = DQN(input_shape, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不训练

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate) # TODO:

    def select_action(self, state, epsilon=0.0):
        """选择动作（根据ε-贪婪策略）

        Args:
            state: 当前状态
            epsilon: 探索率

        Returns:
            选择的动作
        """
        # 获取合法动作
        legal_actions = state.legal_actions()
        if not legal_actions:
            return None

        if np.random.random() < epsilon:
            # 随机探索（只从合法动作中选择）
            return np.random.choice(legal_actions)
        else:
            # 贪婪选择
            with torch.no_grad():
                # 检查state是否为pyspiel.State类型
                if hasattr(state, "observation_tensor"):
                    # 获取观察张量并转换为适当形状
                    obs = np.array(state.observation_tensor(state.current_player()))
                    state = obs.reshape(3, 6, 7)  # 形状: [棋子类型, 行, 列]

                # state = self.flatten_channels(state)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)

                # 创建一个掩码，将非法动作的Q值设为负无穷大
                q_values_np = q_values.cpu().numpy()[0]
                mask = np.ones(self.action_size) * float("-inf")
                mask[legal_actions] = 0
                masked_q_values = q_values_np + mask

                return np.argmax(masked_q_values)

    def get_q_values(self, states):
        """获取一批状态的Q值

        Args:
            states: 状态批次

        Returns:
            预测的Q值
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        return self.policy_net(states_tensor)

    def learn(
        self,
        states,
        actions,
        rewards,
        next_states, 
        q_mcts_values,
        dones,
        lambda_mix=0.5,
    ):
        """从经验中学习

        Args:
            states: 状态批次
            actions: 动作批次
            rewards: 奖励批次
            next_states: 下一状态批次
            q_mcts_values: MCTS估值批次
            dones: 终止标志批次
            lambda_mix: MCTS和DQN目标的混合系数

        Returns:
            当前批次的损失值
        """
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        q_mcts_tensor = torch.FloatTensor(q_mcts_values).to(self.device)
        
        # Double DQN, use next action from policy net and use its Q value in target net
        with torch.no_grad():
            next_actions = self.policy_net(next_states_tensor).argmax(dim=1, keepdim=True)

            next_q_values = self.target_net(next_states_tensor).gather(1, next_actions).squeeze()
            next_q_values[dones] = 0.0
            dqn_targets = rewards_tensor + (self.gamma ** self.n_step) * next_q_values
            # next_q_values = self.target_net(next_states_tensor).max(1)[0]
            # # 对终止状态，下一状态的Q值为0
            # next_q_values = next_q_values * (1 - dones_tensor)

            # # 计算DQN目标
            # dqn_targets = rewards_tensor + self.gamma * next_q_values

            # 混合DQN目标和MCTS估值
            targets = (1 - lambda_mix) * dqn_targets + lambda_mix * q_mcts_tensor

        # 计算当前状态的Q值
        q_values = self.policy_net(states_tensor)
        q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # 计算损失并更新网络
        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """硬更新目标网络（完全复制参数）"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target_network(self):
        """软更新目标网络（部分更新参数）"""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path):
        """保存模型参数"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        """加载模型参数"""
        checkpoint = torch.load(path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # checkpoint = torch.load(path, map_location=torch.device('mps')) #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
    def flatten_channels(self, state: np.array) -> np.array:
        if len(state.shape) > 3: # B, 3, row, col
            state = -state[:, 0, :, :] + state[:, 1, :, :]
        else:
            state = -state[0] + state[1]
        state = np.expand_dims(state, axis=0)
        return state
    


class MCTSDQNAgent(BaseAgent):
    """结合DQNAgent和MCTS的智能体"""

    def __init__(
        self,
        input_shape,
        action_size,
        device="cpu",
        learning_rate=0.001,
        gamma=0.99,
        tau=0.001,
        num_simulations=100,
        uct_c=2.0,
        max_nodes=10000,
        solve=True,
        use_dqn_evaluator=True,
        n_rollouts=1,
        n_step=1
    ):
        """初始化MCTS-DQN智能体

        Args:
            input_shape: 输入状态的形状
            action_size: 动作空间大小
            device: 计算设备 ('cpu' 或 'cuda')
            learning_rate: 学习率
            gamma: 奖励折扣因子
            tau: 目标网络软更新系数
            num_simulations: MCTS搜索的模拟次数
            uct_c: UCT探索常数
            max_nodes: MCTS搜索树最大节点数
            solve: 是否在MCTS中解决终局状态
            use_dqn_evaluator: 是否使用DQN评估器，False则使用随机模拟评估器
            n_rollouts: 随机模拟次数（仅在use_dqn_evaluator=False时使用）
        """
        super(MCTSDQNAgent, self).__init__(input_shape, action_size, device)

        # 创建内部DQNAgent
        self.dqn_agent = DQNAgent(
            input_shape, action_size, device, learning_rate, gamma, tau
        )

        # MCTS参数
        self.num_simulations = num_simulations
        self.uct_c = uct_c
        self.max_nodes = max_nodes
        self.solve = solve
        self.use_dqn_evaluator = use_dqn_evaluator
        self.n_rollouts = n_rollouts
        self.n_step = n_step

        # 当使用mcts_wrapper时，需要初始化游戏
        from open_spiel.python.algorithms import mcts
        import pyspiel
        from utils import get_connect_four_game

        self.game = get_connect_four_game()

        # 创建MCTS包装器
        from mcts_wrapper import MCTSWrapper, DQNEvaluator

        # 根据参数选择评估器
        if use_dqn_evaluator:
            self.evaluator = DQNEvaluator(self.dqn_agent)
        else:
            self.evaluator = mcts.RandomRolloutEvaluator(n_rollouts=n_rollouts)

        self.mcts_wrapper = MCTSWrapper(
            game=self.game,
            num_simulations=num_simulations,
            uct_c=uct_c,
            max_nodes=max_nodes,
            solve=solve,
            use_dqn=use_dqn_evaluator,
            dqn_agent=self.dqn_agent,
        )

        # 手动设置评估器
        self.mcts_wrapper.evaluator = self.evaluator

    def select_action(self, state, epsilon=0.0):
        """选择动作，使用MCTS搜索

        Args:
            state: 当前状态
            epsilon: 探索率（在此实现中不使用，但保留参数以保持接口一致性）

        Returns:
            选择的动作
        """
        # 获取合法动作
        legal_actions = state.legal_actions()
        if not legal_actions:
            return None

        # 如果只有一个合法动作，直接返回
        if len(legal_actions) == 1:
            return legal_actions[0]

        # _ = self.dqn_agent.select_action(state, epsilon)
        if self.num_simulations > 0:
            # 使用MCTS搜索
            best_action, qs, counts, _ = self.mcts_wrapper.search(state)
            # p = "\n".join([f'Col{i}--' + f'C: {counts.get(i)}' + f' Q: {qs.get(i)}' for i in range(7)])
            # print(f'MCTS-q,c:\n{p}')
            # 确保选择的动作是合法的
            if best_action not in legal_actions:
                return np.random.choice(legal_actions)

            return best_action
        else:
            # 直接使用DQN策略，需要传递原始状态
            return self.dqn_agent.select_action(state, epsilon)

    def get_q_values(self, states):
        """获取一批状态的Q值

        Args:
            states: 状态批次

        Returns:
            预测的Q值
        """
        return self.dqn_agent.get_q_values(states)

    def learn(
        self,
        states,
        actions,
        rewards,
        next_states,
        q_mcts_values,
        dones,
        lambda_mix=0.5,
    ):
        """从经验中学习

        Args:
            states: 状态批次
            actions: 动作批次
            rewards: 奖励批次
            next_states: 下一状态批次
            q_mcts_values: MCTS估值批次
            dones: 终止标志批次
            lambda_mix: MCTS和DQN目标的混合系数

        Returns:
            当前批次的损失值
        """
        return self.dqn_agent.learn(
            states, actions, rewards, next_states, q_mcts_values, dones, lambda_mix
        )

    def update_target_network(self):
        """硬更新目标网络"""
        self.dqn_agent.update_target_network()

    def soft_update_target_network(self):
        """软更新目标网络"""
        self.dqn_agent.soft_update_target_network()

    def save(self, path):
        """保存模型参数"""
        self.dqn_agent.save(path)

    def load(self, path):
        """加载模型参数"""
        self.dqn_agent.load(path)

    def set_num_simulations(self, num_simulations):
        """设置MCTS模拟次数

        Args:
            num_simulations: 新的模拟次数
        """
        self.num_simulations = num_simulations
        self.mcts_wrapper.max_simulations = num_simulations
        self.mcts_wrapper.num_simulations = num_simulations

    def set_evaluator(self, use_dqn_evaluator=True, n_rollouts=1):
        """设置MCTS评估器类型

        Args:
            use_dqn_evaluator: 是否使用DQN评估器
            n_rollouts: 随机模拟次数（仅在use_dqn_evaluator=False时使用）
        """
        from open_spiel.python.algorithms import mcts
        from mcts_wrapper import DQNEvaluator

        self.use_dqn_evaluator = use_dqn_evaluator
        self.n_rollouts = n_rollouts

        if use_dqn_evaluator:
            self.evaluator = DQNEvaluator(self.dqn_agent)
        else:
            self.evaluator = mcts.RandomRolloutEvaluator(n_rollouts=n_rollouts)

        # 更新MCTS包装器的评估器
        self.mcts_wrapper.evaluator = self.evaluator
        self.mcts_wrapper.use_dqn = use_dqn_evaluator


class MiniMaxAgent(BaseAgent):
    """基于OpenSpiel的MiniMax算法的智能体"""

    def __init__(self, input_shape, action_size, max_depth, device="cpu"):
        """初始化MiniMax智能体

        Args:
            input_shape: 输入状态的形状
            action_size: 动作空间大小
            device: 计算设备 ('cpu' 或 'cuda')
            max_depth: 最大搜索深度
        """
        super(MiniMaxAgent, self).__init__(input_shape, action_size, device)
        self.max_depth = max_depth

        # 获取游戏实例
        from utils import get_connect_four_game

        self.game = get_connect_four_game()

        # 导入OpenSpiel的AlphaBetaSearch算法
        from open_spiel.python.algorithms import minimax

        self.alpha_beta_search = minimax.alpha_beta_search

        # 设置玩家ID
        self.player_id = 0
        self.opponent_id = 1

    def select_action(self, state, epsilon=0.0):
        """选择动作，使用OpenSpiel的AlphaBetaSearch算法

        Args:
            state: 当前状态
            epsilon: 探索率（在此实现中不使用，但保留参数以保持接口一致性）

        Returns:
            选择的动作
        """
        # 获取合法动作
        legal_actions = state.legal_actions()

        # 如果没有合法动作，返回None
        if not legal_actions:
            return None

        # 如果只有一个合法动作，直接返回
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 设置当前玩家ID
        self.player_id = state.current_player()
        self.opponent_id = 1 if self.player_id == 0 else 0

        # 使用OpenSpiel的AlphaBetaSearch算法
        value, best_action = self.alpha_beta_search(
            game=self.game,
            state=state,
            value_function=self._evaluate_board if self.max_depth > 0 else None,
            maximum_depth=self.max_depth,
        )

        return best_action

    def _evaluate_board(self, state):
        # Observation tensor: shape 2 x 6 x 7 (大部分情况)
        obs = np.array(state.observation_tensor(self.player_id))
        board_channels = obs.reshape(3, 6, 7)
        
        my_pieces = board_channels[1]
        opp_pieces = board_channels[0]

        # Merge into a single 6x7 board
        # 0 = empty
        # 1 = my (player_id) move
        # 2 = opponent's move
        board = np.zeros((6, 7), dtype=int)
        board[my_pieces == 1] = 1
        board[opp_pieces == 1] = 2

        # 现在再去做四连检测
        score = 0

        # 遍历所有可能的 4-length 窗口
        # (Horizontal)
        for row in range(6):
            for col in range(4):
                window = board[row, col : col + 4]
                score += self._evaluate_window(window)

        # (Vertical)
        for row in range(3):
            for col in range(7):
                window = board[row : row + 4, col]
                score += self._evaluate_window(window)

        # (Positive diagonal)
        for row in range(3):
            for col in range(4):
                window = [board[row + i, col + i] for i in range(4)]
                score += self._evaluate_window(window)

        # (Negative diagonal)
        for row in range(3, 6):
            for col in range(4):
                window = [board[row - i, col + i] for i in range(4)]
                score += self._evaluate_window(window)

        # Prioritize center columns
        center_col = board[:, 3]
        score += np.count_nonzero(center_col == 1) * 6

        # Prioritize double threats
        score += self._evaluate_double_threats(board)

        return score

    def _evaluate_double_threats(self, board):
        """评估棋盘上的双重威胁."""
        score = 0
        # (Horizontal)
        for row in range(6):
            for col in range(5):
                window = board[row, col : col + 3]
                score += self._evaluate_threat(window, self.player_id)

        # (Vertical)
        for row in range(4):
            for col in range(7):
                window = board[row : row + 3, col]
                score += self._evaluate_threat(window, self.player_id)

        # (Positive diagonal)
        for row in range(4):
            for col in range(5):
                window = [board[row + i, col + i] for i in range(3)]
                score += self._evaluate_threat(window, self.player_id)

        # (Negative diagonal)
        for row in range(2, 6):
            for col in range(5):
                window = [board[row - i, col + i] for i in range(3)]
                score += self._evaluate_threat(window, self.player_id)
        return score

    def _evaluate_threat(self, window, player):
        player_count = np.count_nonzero(window == player)
        empty_count = np.count_nonzero(window == 0)

        if player_count == 2 and empty_count == 1:
            return 50
        return 0

    def save(self, path):
        """保存智能体参数（MiniMax没有需要保存的模型）"""
        # 保存配置参数
        np.save(path, {"max_depth": self.max_depth})

    def load(self, path):
        """加载智能体参数"""
        # 加载配置参数
        config = np.load(path, allow_pickle=True).item()
        self.max_depth = config["max_depth"]

    def _evaluate_window(self, window):
        """给定4格窗口的打分."""
        player_count = np.count_nonzero(window == 1)
        opp_count = np.count_nonzero(window == 2)
        empty_count = np.count_nonzero(window == 0)

        # 如果我方和对手都在这4格里，都不算潜在连子 => 直接 0 分
        if player_count > 0 and opp_count > 0:
            return 0

        # 我方的各种情况
        if player_count == 4:
            return 1_000_000  # 直接稳赢
        elif player_count == 3 and empty_count == 1:
            return 50
        elif player_count == 2 and empty_count == 2:
            return 10

        # 对手的各种情况（数值上给负分或者更小的值）
        if opp_count == 4:
            return -1_000_000  # 对手马上赢了
        elif opp_count == 3 and empty_count == 1:
            return -100
        elif opp_count == 2 and empty_count == 2:
            return -20

        return 0
