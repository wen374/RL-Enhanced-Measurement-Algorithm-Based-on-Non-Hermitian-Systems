#!/usr/bin/env python3
"""
非厄米奇异点强化学习优化系统
Exceptional Points (EPs) Optimization via Reinforcement Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SystemParameters:
    """系统参数配置"""
    gamma: float = 10.0  # 损耗参数
    pump_power: float = 50.0  # 泵浦功率
    detuning: float = 0.0  # 失谐
    coupling: float = 0.5  # 耦合强度
    omega0: float = 100.0  # 基础频率

class NonHermitianSystem:
    """非厄米系统 - Faraday FP腔数字孪生"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
    def calculate_eigenfrequencies(self) -> Tuple[float, float]:
        """
        计算2x2非厄米哈密顿矩阵的本征频率
        H = [[ω0 + δ + i(g - γ), κ],
             [κ, ω0 - δ + i(g - γ)]]
        """
        ω0 = self.params.omega0
        δ = self.params.detuning
        γ = self.params.gamma
        g = self.params.pump_power
        κ = self.params.coupling
        
        # 构建哈密顿矩阵 (实部和虚部分离)
        H11_re = ω0 + δ
        H11_im = g - γ
        H22_re = ω0 - δ
        H22_im = g - γ
        H12 = κ
        H21 = κ
        
        # 计算本征值 (只考虑实部用于简化)
        trace_re = (H11_re + H22_re) / 2
        diff_re = (H11_re - H22_re) / 2
        diff_im = (H11_im - H22_im) / 2
        
        # 判别式
        discriminant = diff_re**2 - diff_im**2 + H12 * H21
        
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            lambda1 = trace_re + sqrt_disc
            lambda2 = trace_re - sqrt_disc
        else:
            # 复数判别式情况
            sqrt_disc = np.sqrt(abs(discriminant))
            lambda1 = trace_re
            lambda2 = trace_re
            
        return lambda1, lambda2
    
    def get_ep_distance(self) -> float:
        """计算到奇异点的距离（本征频率简并度）"""
        λ1, λ2 = self.calculate_eigenfrequencies()
        return abs(λ1 - λ2)
    
    def get_transmission_peaks(self) -> Tuple[float, float]:
        """计算传输谱峰值"""
        λ1, λ2 = self.calculate_eigenfrequencies()
        gain_loss_ratio = self.params.pump_power / max(self.params.gamma, 0.1)
        
        t1 = gain_loss_ratio * np.exp(-abs(λ1 - self.params.omega0) / 10)
        t2 = gain_loss_ratio * np.exp(-abs(λ2 - self.params.omega0) / 10)
        
        return t1, t2
    
    def get_sensitivity_enhancement(self) -> float:
        """计算测量灵敏度增强因子"""
        ep_dist = self.get_ep_distance()
        # 在EP点附近，灵敏度增强
        sensitivity = 1.0 / (ep_dist + 0.01)  # 避免除零
        return min(sensitivity, 100.0)  # 限制最大值

class RLAgent:
    """强化学习代理 - Q-Learning"""
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q表 - 使用字典存储状态-动作值
        self.q_table = {}
        
        # 动作空间定义
        self.actions = [
            ('increase_gamma', np.array([1.0, 0, 0])),
            ('decrease_gamma', np.array([-1.0, 0, 0])),
            ('increase_pump', np.array([0, 2.0, 0])),
            ('decrease_pump', np.array([0, -2.0, 0])),
            ('increase_detuning', np.array([0, 0, 0.5])),
            ('decrease_detuning', np.array([0, 0, -0.5])),
            ('fine_tune_gamma', np.array([0.2, 0, 0])),
            ('fine_tune_pump', np.array([0, 0.5, 0])),
            ('no_action', np.array([0, 0, 0]))
        ]
        
        self.n_actions = len(self.actions)
        
    def discretize_state(self, params: SystemParameters, ep_distance: float) -> str:
        """将连续状态离散化"""
        γ_bin = int(params.gamma / 2)
        p_bin = int(params.pump_power / 10)
        d_bin = int(params.detuning / 1)
        ep_bin = int(ep_distance * 10)
        
        return f"{γ_bin}_{p_bin}_{d_bin}_{ep_bin}"
    
    def get_q_value(self, state: str, action_idx: int) -> float:
        """获取Q值"""
        key = f"{state}_{action_idx}"
        return self.q_table.get(key, 0.0)
    
    def set_q_value(self, state: str, action_idx: int, value: float):
        """设置Q值"""
        key = f"{state}_{action_idx}"
        self.q_table[key] = value
    
    def select_action(self, state: str) -> Tuple[int, np.ndarray]:
        """ε-greedy策略选择动作"""
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            # 利用：选择Q值最大的动作
            q_values = [self.get_q_value(state, i) for i in range(self.n_actions)]
            action_idx = np.argmax(q_values)
        
        return action_idx, self.actions[action_idx][1]
    
    def update_q_table(self, state: str, action_idx: int, 
                      reward: float, next_state: str):
        """Q-Learning更新"""
        current_q = self.get_q_value(state, action_idx)
        
        # 找到下一个状态的最大Q值
        next_q_values = [self.get_q_value(next_state, i) for i in range(self.n_actions)]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Q-Learning更新公式
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.set_q_value(state, action_idx, new_q)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class RewardFunction:
    """奖励函数设计"""
    
    @staticmethod
    def calculate(ep_distance: float, 
                 prev_ep_distance: float,
                 transmission: Tuple[float, float],
                 sensitivity: float) -> float:
        """
        计算奖励值
        目标：最小化EP距离，最大化传输和灵敏度
        """
        # EP距离奖励（负奖励，距离越小越好）
        ep_reward = -ep_distance * 10
        
        # 改进奖励（鼓励向EP点靠近）
        improvement_reward = (prev_ep_distance - ep_distance) * 50
        
        # 传输增强奖励
        transmission_reward = np.mean(transmission) * 5
        
        # 灵敏度奖励
        sensitivity_reward = np.log1p(sensitivity) * 10
        
        # EP点奖励（达到EP点给予大额奖励）
        ep_bonus = 100 if ep_distance < 0.1 else 0
        
        # 稳定性惩罚（避免参数过大波动）
        stability_penalty = 0
        
        total_reward = (ep_reward + improvement_reward + 
                       transmission_reward + sensitivity_reward + 
                       ep_bonus - stability_penalty)
        
        return total_reward

class EPOptimizationEnvironment:
    """EP优化环境"""
    
    def __init__(self):
        self.params = SystemParameters()
        self.system = NonHermitianSystem(self.params)
        self.agent = RLAgent()
        self.reward_func = RewardFunction()
        
        # 历史记录
        self.ep_distance_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.sensitivity_history = deque(maxlen=1000)
        self.param_history = deque(maxlen=1000)
        
        # 最佳配置
        self.best_params = None
        self.best_ep_distance = float('inf')
        
        # 统计信息
        self.episode = 0
        self.total_reward = 0
        
    def reset(self):
        """重置环境"""
        self.params = SystemParameters()
        self.system = NonHermitianSystem(self.params)
        self.episode = 0
        self.total_reward = 0
        
    def step(self) -> dict:
        """执行一步优化"""
        # 获取当前状态
        ep_distance = self.system.get_ep_distance()
        state = self.agent.discretize_state(self.params, ep_distance)
        
        # 选择动作
        action_idx, action_delta = self.agent.select_action(state)
        
        # 保存当前EP距离
        prev_ep_distance = ep_distance
        
        # 执行动作，更新参数
        self._update_parameters(action_delta)
        
        # 计算新状态
        new_ep_distance = self.system.get_ep_distance()
        transmission = self.system.get_transmission_peaks()
        sensitivity = self.system.get_sensitivity_enhancement()
        
        next_state = self.agent.discretize_state(self.params, new_ep_distance)
        
        # 计算奖励
        reward = self.reward_func.calculate(
            new_ep_distance, prev_ep_distance, transmission, sensitivity
        )
        
        # 更新Q表
        self.agent.update_q_table(state, action_idx, reward, next_state)
        
        # 更新统计
        self.total_reward += reward
        self.episode += 1
        
        # 记录历史
        self.ep_distance_history.append(new_ep_distance)
        self.reward_history.append(reward)
        self.sensitivity_history.append(sensitivity)
        self.param_history.append({
            'gamma': self.params.gamma,
            'pump_power': self.params.pump_power,
            'detuning': self.params.detuning
        })
        
        # 更新最佳配置
        if new_ep_distance < self.best_ep_distance:
            self.best_ep_distance = new_ep_distance
            self.best_params = SystemParameters(
                gamma=self.params.gamma,
                pump_power=self.params.pump_power,
                detuning=self.params.detuning
            )
            logger.info(f"New best EP distance: {self.best_ep_distance:.6f}")
        
        # 衰减探索率
        if self.episode % 10 == 0:
            self.agent.decay_epsilon()
        
        return {
            'episode': self.episode,
            'ep_distance': new_ep_distance,
            'reward': reward,
            'total_reward': self.total_reward,
            'sensitivity': sensitivity,
            'epsilon': self.agent.epsilon
        }
    
    def _update_parameters(self, action_delta: np.ndarray):
        """更新系统参数"""
        # 应用动作并限制参数范围
        self.params.gamma = np.clip(
            self.params.gamma + action_delta[0], 0.1, 20.0
        )
        self.params.pump_power = np.clip(
            self.params.pump_power + action_delta[1], 1.0, 100.0
        )
        self.params.detuning = np.clip(
            self.params.detuning + action_delta[2], -10.0, 10.0
        )
        
        # 更新系统
        self.system.params = self.params
    
    def train(self, n_episodes: int = 1000):
        """训练循环"""
        logger.info(f"Starting training for {n_episodes} episodes...")
        
        for _ in range(n_episodes):
            info = self.step()
            
            if self.episode % 100 == 0:
                logger.info(
                    f"Episode {info['episode']}: "
                    f"EP Distance={info['ep_distance']:.6f}, "
                    f"Reward={info['reward']:.2f}, "
                    f"Sensitivity={info['sensitivity']:.2f}, "
                    f"ε={info['epsilon']:.3f}"
                )
        
        logger.info(f"Training completed. Best EP distance: {self.best_ep_distance:.6f}")
        if self.best_params:
            logger.info(f"Best parameters: γ={self.best_params.gamma:.2f}, "
                       f"P={self.best_params.pump_power:.2f}, "
                       f"δ={self.best_params.detuning:.2f}")

class Visualizer:
    """可视化工具"""
    
    def __init__(self, env: EPOptimizationEnvironment):
        self.env = env
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Non-Hermitian EP Optimization via Reinforcement Learning', fontsize=16)
        
    def plot_results(self):
        """绘制训练结果"""
        episodes = range(len(self.env.ep_distance_history))
        
        # EP距离演化
        ax1 = self.axes[0, 0]
        ax1.clear()
        ax1.plot(episodes, self.env.ep_distance_history, 'b-', alpha=0.7)
        ax1.axhline(y=0.1, color='g', linestyle='--', label='EP Threshold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('EP Distance')
        ax1.set_title('Eigenfrequency Degeneracy Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 奖励历史
        ax2 = self.axes[0, 1]
        ax2.clear()
        ax2.plot(episodes, self.env.reward_history, 'r-', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.set_title('Training Reward')
        ax2.grid(True, alpha=0.3)
        
        # 灵敏度增强
        ax3 = self.axes[0, 2]
        ax3.clear()
        ax3.plot(episodes, self.env.sensitivity_history, 'g-', alpha=0.7)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Sensitivity Enhancement')
        ax3.set_title('Measurement Sensitivity')
        ax3.grid(True, alpha=0.3)
        
        # 参数演化
        if self.env.param_history:
            params = list(self.env.param_history)
            gamma_vals = [p['gamma'] for p in params]
            pump_vals = [p['pump_power'] for p in params]
            detuning_vals = [p['detuning'] for p in params]
            
            ax4 = self.axes[1, 0]
            ax4.clear()
            ax4.plot(episodes, gamma_vals, 'b-', label='γ (Loss)', alpha=0.7)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Loss Parameter γ')
            ax4.set_title('Loss Parameter Evolution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            ax5 = self.axes[1, 1]
            ax5.clear()
            ax5.plot(episodes, pump_vals, 'r-', label='Pump Power', alpha=0.7)
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Pump Power')
            ax5.set_title('Pump Power Evolution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            ax6 = self.axes[1, 2]
            ax6.clear()
            ax6.plot(episodes, detuning_vals, 'g-', label='Detuning δ', alpha=0.7)
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Detuning δ')
            ax6.set_title('Detuning Evolution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def animate_training(self, interval: int = 100):
        """实时动画显示训练过程"""
        def update(frame):
            if self.env.episode > 0:
                self.plot_results()
            return self.axes.ravel()
        
        ani = FuncAnimation(self.fig, update, interval=interval, blit=False)
        return ani
    
    def show(self):
        """显示图表"""
        plt.show()

def main():
    """主函数"""
    # 创建环境
    env = EPOptimizationEnvironment()
    
    # 训练
    env.train(n_episodes=1000)
    
    # 可视化结果
    viz = Visualizer(env)
    viz.plot_results()
    viz.show()
    
    # 输出最终结果
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best EP Distance: {env.best_ep_distance:.6f}")
    if env.best_params:
        print(f"Optimal Parameters:")
        print(f"  - Loss (γ): {env.best_params.gamma:.2f}")
        print(f"  - Pump Power: {env.best_params.pump_power:.2f}")
        print(f"  - Detuning (δ): {env.best_params.detuning:.2f}")
        print(f"  - Coupling (κ): {env.best_params.coupling:.2f}")
    
    # 计算最终灵敏度增强
    final_system = NonHermitianSystem(env.best_params)
    final_sensitivity = final_system.get_sensitivity_enhancement()
    print(f"\nFinal Sensitivity Enhancement: {final_sensitivity:.2f}x")
    
    return env

if __name__ == "__main__":
    main()
