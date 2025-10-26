#!/usr/bin/env python3
"""
高级EP优化模块 - 深度强化学习和并行训练
Advanced EP Optimization with Deep Reinforcement Learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import pickle
import json
from dataclasses import dataclass, asdict

# 经验回放缓冲区
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_dim: int = 4, hidden_dim: int = 256, 
                 action_dim: int = 9):
        super(DQN, self).__init__()
        
        # 全连接层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, 
                 state_dim: int = 4,
                 action_dim: int = 9,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_size: int = 10000,
                 batch_size: int = 64):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # 神经网络
        self.q_network = DQN(state_dim, 256, action_dim)
        self.target_network = DQN(state_dim, 256, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=buffer_size)
        
        # 更新目标网络
        self.update_target_network()
        
        # 动作空间
        self.actions = [
            np.array([1.0, 0, 0]),    # increase gamma
            np.array([-1.0, 0, 0]),   # decrease gamma
            np.array([0, 2.0, 0]),     # increase pump
            np.array([0, -2.0, 0]),    # decrease pump
            np.array([0, 0, 0.5]),     # increase detuning
            np.array([0, 0, -0.5]),    # decrease detuning
            np.array([0.2, 0.5, 0]),   # fine tune gamma & pump
            np.array([-0.2, -0.5, 0]), # fine tune gamma & pump
            np.array([0, 0, 0])        # no action
        ]
    
    def state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """将状态转换为张量"""
        return torch.FloatTensor(state).unsqueeze(0)
    
    def select_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """ε-greedy动作选择"""
        if random.random() < self.epsilon:
            # 探索
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # 利用
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        return action_idx, self.actions[action_idx]
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

class AdvancedEPEnvironment:
    """高级EP优化环境"""
    
    def __init__(self, use_dqn: bool = True):
        self.use_dqn = use_dqn
        self.reset()
        
        # 创建智能体
        if use_dqn:
            self.agent = DQNAgent()
        else:
            from ep_rl_optimization import RLAgent
            self.agent = RLAgent()
        
        # 历史记录
        self.history = {
            'episodes': [],
            'ep_distances': [],
            'rewards': [],
            'sensitivities': [],
            'parameters': []
        }
    
    def reset(self):
        """重置环境"""
        self.gamma = 10.0
        self.pump_power = 50.0
        self.detuning = 0.0
        self.coupling = 0.5
        self.omega0 = 100.0
        self.episode = 0
        self.total_reward = 0
        self.best_ep_distance = float('inf')
        
    def get_state(self) -> np.ndarray:
        """获取当前状态向量"""
        ep_distance = self.calculate_ep_distance()
        state = np.array([
            self.gamma / 20.0,  # 归一化
            self.pump_power / 100.0,
            (self.detuning + 10) / 20.0,
            ep_distance
        ])
        return state
    
    def calculate_eigenfrequencies(self) -> Tuple[float, float]:
        """计算本征频率"""
        # 构建哈密顿矩阵
        H11_re = self.omega0 + self.detuning
        H11_im = self.pump_power - self.gamma
        H22_re = self.omega0 - self.detuning
        H22_im = self.pump_power - self.gamma
        
        # 计算本征值
        trace_re = (H11_re + H22_re) / 2
        diff_re = (H11_re - H22_re) / 2
        diff_im = (H11_im - H22_im) / 2
        
        discriminant = diff_re**2 - diff_im**2 + self.coupling**2
        
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            lambda1 = trace_re + sqrt_disc
            lambda2 = trace_re - sqrt_disc
        else:
            sqrt_disc = np.sqrt(abs(discriminant))
            lambda1 = trace_re
            lambda2 = trace_re
            
        return lambda1, lambda2
    
    def calculate_ep_distance(self) -> float:
        """计算EP距离"""
        lambda1, lambda2 = self.calculate_eigenfrequencies()
        return abs(lambda1 - lambda2)
    
    def calculate_sensitivity(self) -> float:
        """计算灵敏度增强"""
        ep_distance = self.calculate_ep_distance()
        sensitivity = 1.0 / (ep_distance + 0.01)
        return min(sensitivity, 100.0)
    
    def calculate_reward(self, ep_distance: float, prev_ep_distance: float) -> float:
        """计算奖励"""
        # EP距离奖励
        ep_reward = -ep_distance * 10
        
        # 改进奖励
        improvement_reward = (prev_ep_distance - ep_distance) * 50
        
        # 灵敏度奖励
        sensitivity = self.calculate_sensitivity()
        sensitivity_reward = np.log1p(sensitivity) * 10
        
        # EP点奖励
        ep_bonus = 100 if ep_distance < 0.1 else 0
        
        # 参数稳定性惩罚
        param_penalty = 0
        if self.gamma < 1 or self.gamma > 19:
            param_penalty -= 10
        if self.pump_power < 5 or self.pump_power > 95:
            param_penalty -= 10
            
        total_reward = ep_reward + improvement_reward + sensitivity_reward + ep_bonus + param_penalty
        
        return total_reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作"""
        # 保存当前EP距离
        prev_ep_distance = self.calculate_ep_distance()
        
        # 执行动作
        self.gamma = np.clip(self.gamma + action[0], 0.1, 20.0)
        self.pump_power = np.clip(self.pump_power + action[1], 1.0, 100.0)
        self.detuning = np.clip(self.detuning + action[2], -10.0, 10.0)
        
        # 计算新状态
        new_state = self.get_state()
        ep_distance = self.calculate_ep_distance()
        
        # 计算奖励
        reward = self.calculate_reward(ep_distance, prev_ep_distance)
        self.total_reward += reward
        
        # 检查是否完成
        done = ep_distance < 0.01 or self.episode >= 1000
        
        # 更新最佳记录
        if ep_distance < self.best_ep_distance:
            self.best_ep_distance = ep_distance
            self.best_params = {
                'gamma': self.gamma,
                'pump_power': self.pump_power,
                'detuning': self.detuning
            }
        
        # 记录历史
        self.history['episodes'].append(self.episode)
        self.history['ep_distances'].append(ep_distance)
        self.history['rewards'].append(reward)
        self.history['sensitivities'].append(self.calculate_sensitivity())
        self.history['parameters'].append({
            'gamma': self.gamma,
            'pump_power': self.pump_power,
            'detuning': self.detuning
        })
        
        self.episode += 1
        
        info = {
            'ep_distance': ep_distance,
            'sensitivity': self.calculate_sensitivity(),
            'total_reward': self.total_reward,
            'best_ep_distance': self.best_ep_distance
        }
        
        return new_state, reward, done, info
    
    def train_dqn(self, episodes: int = 1000):
        """使用DQN训练"""
        for episode in range(episodes):
            state = self.get_state()
            episode_reward = 0
            
            for step in range(200):
                # 选择动作
                action_idx, action = self.agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = self.step(action)
                
                # 存储经验
                self.agent.remember(state, action_idx, reward, next_state, done)
                
                # 训练
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # 更新目标网络
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            # 衰减探索率
            self.agent.decay_epsilon()
            
            # 打印进度
            if episode % 100 == 0:
                print(f"Episode {episode}: EP Distance={info['ep_distance']:.6f}, "
                      f"Reward={episode_reward:.2f}, ε={self.agent.epsilon:.3f}")
        
        return self.history

class ParallelTrainer:
    """并行训练器"""
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or cpu_count()
    
    def train_worker(self, args):
        """单个工作进程的训练"""
        worker_id, episodes = args
        env = AdvancedEPEnvironment(use_dqn=True)
        
        # 随机初始化参数以增加多样性
        env.gamma = np.random.uniform(5, 15)
        env.pump_power = np.random.uniform(30, 70)
        
        history = env.train_dqn(episodes)
        
        return {
            'worker_id': worker_id,
            'best_ep_distance': env.best_ep_distance,
            'best_params': env.best_params if hasattr(env, 'best_params') else None,
            'history': history
        }
    
    def train_parallel(self, total_episodes: int = 4000):
        """并行训练多个智能体"""
        episodes_per_worker = total_episodes // self.n_workers
        
        # 创建工作任务
        tasks = [(i, episodes_per_worker) for i in range(self.n_workers)]
        
        # 并行训练
        with Pool(self.n_workers) as pool:
            results = pool.map(self.train_worker, tasks)
        
        # 找到最佳结果
        best_result = min(results, key=lambda x: x['best_ep_distance'])
        
        print(f"\n并行训练完成!")
        print(f"最佳EP距离: {best_result['best_ep_distance']:.6f}")
        if best_result['best_params']:
            print(f"最佳参数: {best_result['best_params']}")
        
        return results

class AdaptiveController:
    """自适应控制器 - 用于实时EP保持"""
    
    def __init__(self, target_ep_distance: float = 0.1):
        self.target_ep_distance = target_ep_distance
        self.pid_gains = {'p': 10.0, 'i': 0.1, 'd': 1.0}
        self.integral = 0
        self.last_error = 0
        
    def control_step(self, current_ep_distance: float, 
                    current_params: dict) -> dict:
        """PID控制步骤"""
        # 计算误差
        error = current_ep_distance - self.target_ep_distance
        
        # PID计算
        self.integral += error
        derivative = error - self.last_error
        
        control_signal = (self.pid_gains['p'] * error + 
                         self.pid_gains['i'] * self.integral + 
                         self.pid_gains['d'] * derivative)
        
        # 更新参数
        new_params = current_params.copy()
        
        # 根据控制信号调整参数
        if abs(control_signal) > 0.1:
            # 主要调整gamma和pump_power
            new_params['gamma'] += control_signal * 0.1
            new_params['pump_power'] -= control_signal * 0.2
            
            # 限制参数范围
            new_params['gamma'] = np.clip(new_params['gamma'], 0.1, 20.0)
            new_params['pump_power'] = np.clip(new_params['pump_power'], 1.0, 100.0)
        
        self.last_error = error
        
        return new_params

def test_advanced_system():
    """测试高级系统"""
    print("="*60)
    print("非厄米奇异点高级优化系统测试")
    print("="*60)
    
    # 1. 测试DQN
    print("\n1. 测试DQN训练...")
    env = AdvancedEPEnvironment(use_dqn=True)
    history = env.train_dqn(episodes=500)
    print(f"DQN训练完成: 最佳EP距离 = {env.best_ep_distance:.6f}")
    
    # 2. 测试并行训练
    print("\n2. 测试并行训练...")
    trainer = ParallelTrainer(n_workers=4)
    results = trainer.train_parallel(total_episodes=400)
    
    # 3. 测试自适应控制
    print("\n3. 测试自适应控制器...")
    controller = AdaptiveController(target_ep_distance=0.05)
    
    current_params = {'gamma': 10.0, 'pump_power': 50.0, 'detuning': 0.0}
    for i in range(10):
        # 模拟EP距离
        ep_distance = np.random.uniform(0.01, 0.5)
        new_params = controller.control_step(ep_distance, current_params)
        print(f"Step {i}: EP距离={ep_distance:.4f}, "
              f"新参数: γ={new_params['gamma']:.2f}, P={new_params['pump_power']:.2f}")
        current_params = new_params
    
    print("\n测试完成!")
    
    return env, results

if __name__ == "__main__":
    test_advanced_system()
