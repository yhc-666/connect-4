import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

class MCTSWrapper:
    """OpenSpiel MCTS算法的包装器"""
    
    def __init__(self, game, num_simulations=100, uct_c=2.0, max_nodes=10000, dirichlet_alpha=0.3, 
                 dirichlet_noise=False, solve=True, seed=None):
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
        """
        self.game = game
        self.num_simulations = num_simulations
        self.uct_c = uct_c
        self.max_simulations = num_simulations
        self.max_nodes = max_nodes
        self.solve = solve
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)
        
        # 创建MCTS评估器
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
            best_action = max(root.children, key=lambda c: c.explore_count).action
        else:
            # 如果没有子节点，从合法动作中随机选择
            best_action = np.random.choice(state.legal_actions())
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
        return obs.reshape(3, 6, 7)  # Connect4的观察空间形状：[棋子类型, 行, 列] 