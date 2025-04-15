import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
import torch

class DQNEvaluator(mcts.Evaluator):
    """
    使用DQN评估状态价值的评估器
    用于MCTS搜索中评估叶子节点
    """
    
    def __init__(self, dqn_agent):
        """初始化DQN评估器
        
        Args:
            dqn_agent: DQN智能体实例，用于评估状态
        """
        self.dqn_agent = dqn_agent
        
    def evaluate(self, state):
        """使用DQN评估状态
        
        Args:
            state: 游戏状态
            
        Returns:
            所有玩家的估值
        """
        player = state.current_player()
        # 获取状态表示
        state_repr = MCTSWrapper.get_state_representation(state, player)
        
        # 使用DQN获取Q值
        with torch.no_grad():
            q_values = self.dqn_agent.get_q_values(np.expand_dims(state_repr, axis=0)).cpu().numpy()
            q_values = q_values[0]
        
        # 取所有动作中的最大Q值作为价值估计
        max_q_value = max(q_values) if len(q_values) > 0 else 0.0
        # print(max_q_value) 
        # 对于Connect4这样的零和游戏，一个玩家的收益是另一个玩家的损失
        return [max_q_value, -max_q_value] if player == 0 else [-max_q_value, max_q_value]
    
    def prior(self, state):
        """根据DQN估值为动作分配先验概率
        
        Args:
            state: 游戏状态
            
        Returns:
            动作及其先验概率
        """
        if state.is_chance_node():
            return state.chance_outcomes()
        
        player = state.current_player()
        legal_actions = state.legal_actions()
        
        if not legal_actions:
            return []
        
        # 获取状态表示
        state_repr = MCTSWrapper.get_state_representation(state, player)
        
        # 使用DQN获取Q值
        with torch.no_grad():
            q_values = self.dqn_agent.get_q_values(np.expand_dims(state_repr, axis=0)).cpu().numpy()[0]
        
        # 只考虑合法动作的Q值
        legal_q_values = [q_values[action] for action in legal_actions]
        
        # 使用softmax将Q值转换为概率
        exp_values = np.exp(legal_q_values - np.max(legal_q_values))  # 减去最大值避免溢出
        probs = exp_values / np.sum(exp_values)
        
        return [(action, prob) for action, prob in zip(legal_actions, probs)]

class MCTSWrapper:
    """OpenSpiel MCTS算法的包装器"""
    
    def __init__(self, game, num_simulations=100, uct_c=2.0, max_nodes=10000, dirichlet_alpha=0.3, 
                 dirichlet_eps=0.25, solve=True, use_dqn=False, dqn_agent=None, test=False):
        """初始化MCTS搜索
        
        Args:
            game: OpenSpiel游戏实例
            num_simulations: 每步MCTS搜索的模拟次数
            uct_c: UCT探索常数
            max_nodes: MCTS搜索树最大节点数
            dirichlet_alpha: Dirichlet噪声参数
            dirichlet_noise: 是否添加Dirichlet噪声（用于根节点探索）
            solve: 是否在MCTS中解决终局状态
            seed: 随机数种子
            use_dqn: 是否使用DQN评估叶子节点
            dqn_agent: DQN智能体实例
        """
        self.game = game
        self.num_simulations = num_simulations
        self.uct_c = uct_c
        self.max_simulations = num_simulations
        self.max_nodes = max_nodes
        self.solve = solve
        self.use_dqn = use_dqn
        self.dirichlet_noise = None
        if not test:
            self.dirichlet_noise = (dirichlet_eps, dirichlet_alpha)
        
        # 创建评估器
        if use_dqn and dqn_agent is not None:
            # 使用DQN评估器
            self.evaluator = DQNEvaluator(dqn_agent)
        else:
            # 使用随机模拟评估器
            self.evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)
        
    def create_bot(self):
        """创建一个新的MCTS机器人实例"""
        return mcts.MCTSBot(
            game=self.game,
            uct_c=self.uct_c,
            max_simulations=self.max_simulations,
            evaluator=self.evaluator,
            solve=self.solve,
            verbose=False,
            dirichlet_noise=self.dirichlet_noise
        )
    
    def search(self, state):
        """执行MCTS搜索并返回动作及其Q值
        
        Args:
            state: 当前游戏状态
            
        Returns:
            最佳动作，所有动作的Q值，访问计数和先验概率
        """
        bot = self.create_bot()
        
        # 执行搜索并获取搜索树
        root = bot.mcts_search(state)

        # 获取所有动作的Q值和访问计数
        action_q_values = {}
        action_counts = {}
        action_priors = {}
        
        for child in root.children:
            action = child.action
            q_value = child.total_reward / max(child.explore_count, 1)
            visit_count = child.explore_count
            prior = child.prior
            
            action_q_values[action] = q_value
            action_counts[action] = visit_count
            action_priors[action] = prior
        
        # 选择最佳动作（访问次数最多的动作）
        if root.children:
            best_action = root.best_child().action
            # most_visited = max(c.explore_count for c in root.children)
            # best_action = max(
            #     (c for c in root.children if c.explore_count == most_visited),
            #     key=lambda c: c.total_reward,
            # ).action
        else:
            # 如果没有子节点，从合法动作中随机选择
            best_action = np.random.choice(state.legal_actions()).item()
            action_q_values[best_action] = 0.0
            action_counts[best_action] = 0
            action_priors[best_action] = 1.0
        
        return best_action, action_q_values, action_counts, action_priors
    
    @staticmethod
    def get_state_representation(state, player_id):
        """将游戏状态转换为神经网络输入格式
        
        Args:
            state: OpenSpiel游戏状态
            player_id: 玩家ID
        
        Returns:
            状态表示为张量，形状为 [通道, 行, 列]
        """
        # 对于Connect4，我们返回观察张量并重新排列维度
        obs = np.array(state.observation_tensor(player_id))
        obs = obs.reshape(3, 6, 7)
        # obs = -obs[0] + obs[1]
        # obs = obs.reshape(1,6,7)
        return obs