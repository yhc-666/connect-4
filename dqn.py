import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        self.conv1 = nn.Conv2d(input_shape[0], 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 计算卷积层输出尺寸
        conv_output_size = 256 * input_shape[1] * input_shape[2]
        
        # 注意力层
        self.attention_conv = nn.Conv2d(256, 1, kernel_size=1)
        
        # Q值预测头
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, action_size)
        
        # 价值预测头（辅助任务）
        self.value_fc = nn.Linear(512, 1)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入状态表示，形状为 (batch_size, channels, rows, cols)
            
        Returns:
            每个动作的Q值预测
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        features = F.relu(self.bn4(self.conv4(x)))
        
        # 应用注意力机制
        attention_weights = F.softmax(self.attention_conv(features).view(features.size(0), -1), dim=1)
        attention_weights = attention_weights.view(attention_weights.size(0), 1, features.size(2), features.size(3))
        attention_applied = features * attention_weights
        
        # 计算输出尺寸 (不再拼接特征，以避免维度问题)
        x = features + attention_applied  # 使用加法而不是拼接
        x = x.view(x.size(0), -1)  # 展平
        
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        
        # 保存这个特征表示用于价值预测
        features_fc = x
        
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        action_values = self.fc3(x)
        
        # 价值预测（可选，仅在训练期间使用）
        state_value = self.value_fc(features_fc)
        
        return action_values


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, input_shape, action_size, device='cpu', learning_rate=0.001, gamma=0.99, tau=0.001):
        """初始化DQN智能体
        
        Args:
            input_shape: 输入状态的形状
            action_size: 动作空间大小
            device: 计算设备 ('cpu' 或 'cuda')
            learning_rate: 学习率
            gamma: 奖励折扣因子
            tau: 目标网络软更新系数
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = torch.device(device)
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 目标网络软更新系数
        
        # 创建主网络和目标网络
        self.policy_net = DQN(input_shape, action_size).to(self.device)
        self.target_net = DQN(input_shape, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不训练
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
    def select_action(self, state, epsilon=0.0, legal_actions=None):
        """选择动作（根据ε-贪婪策略）
        
        Args:
            state: 当前状态
            epsilon: 探索概率
            legal_actions: 可选的合法动作列表，如果提供，将只从这些动作中选择
            
        Returns:
            选择的动作
        """
        if np.random.random() < epsilon:
            # 随机探索
            if legal_actions is not None:
                return np.random.choice(legal_actions)
            else:
                return np.random.randint(self.action_size)
        else:
            # 贪婪选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                if legal_actions is not None:
                    # 仅考虑合法动作
                    temp_q_values = q_values.cpu().numpy()[0]
                    masked_q_values = np.full(temp_q_values.shape, float('-inf'))
                    for action in legal_actions:
                        masked_q_values[action] = temp_q_values[action]
                    return np.argmax(masked_q_values)
                else:
                    return q_values.argmax().item()
    
    def get_q_values(self, states):
        """获取一批状态的Q值
        
        Args:
            states: 状态批次
            
        Returns:
            预测的Q值
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        return self.policy_net(states_tensor)
    
    def learn(self, states, actions, rewards, next_states, q_mcts_values, dones, lambda_mix=0.5):
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
        
        # 计算当前状态的Q值
        q_values = self.policy_net(states_tensor)
        q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # 计算下一状态的最大Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor).max(1)[0]
            # 对终止状态，下一状态的Q值为0
            next_q_values = next_q_values * (1 - dones_tensor)
            
            # 计算DQN目标
            dqn_targets = rewards_tensor + self.gamma * next_q_values
            
            # 混合DQN目标和MCTS估值
            targets = (1 - lambda_mix) * dqn_targets + lambda_mix * q_mcts_tensor
        
        # 计算损失并更新网络
        loss = F.mse_loss(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """硬更新目标网络（完全复制参数）"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def soft_update_target_network(self):
        """软更新目标网络（部分更新参数）"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path):
        """保存模型参数"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """加载模型参数"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 