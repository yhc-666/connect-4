import numpy as np
import random
from collections import namedtuple, deque

# 定义经验元组的结构
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'q_mcts', 'done'])

class ReplayBuffer:
    """经验回放缓冲区，存储并采样游戏经验"""
    
    def __init__(self, capacity):
        """初始化回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, q_mcts, done):
        """添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            q_mcts: MCTS搜索得到的Q值
            done: 是否游戏结束
        """
        experience = Experience(state, action, reward, next_state, q_mcts, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """从缓冲区中随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            包含状态、动作、奖励、下一个状态、MCTS的Q值和结束标志的批次
        """
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        q_mcts_values = np.array([e.q_mcts for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, q_mcts_values, dones
    
    def __len__(self):
        """返回缓冲区中的经验数量"""
        return len(self.buffer) 