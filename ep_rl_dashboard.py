#!/usr/bin/env python3
"""
EP优化系统实时监控界面
Real-time Monitoring Dashboard for EP Optimization
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from threading import Thread, Lock
import time
from ep_rl_optimization import (
    EPOptimizationEnvironment, 
    SystemParameters,
    NonHermitianSystem
)

class RealTimeDashboard:
    """实时监控仪表盘"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("EP强化学习优化系统 - 实时监控")
        self.root.geometry("1400x900")
        
        # 环境和状态
        self.env = EPOptimizationEnvironment()
        self.is_running = False
        self.lock = Lock()
        
        # 数据缓冲区
        self.max_points = 500
        self.episode_data = []
        self.ep_distance_data = []
        self.reward_data = []
        self.sensitivity_data = []
        self.gamma_data = []
        self.pump_data = []
        self.detuning_data = []
        
        # 创建UI
        self._create_widgets()
        self._create_plots()
        
        # 更新循环
        self.update_interval = 100  # ms
        self.update_plots()
        
    def _create_widgets(self):
        """创建UI组件"""
        # 顶部控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 控制按钮
        self.start_button = ttk.Button(
            control_frame, text="开始训练", command=self.start_training
        )
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(
            control_frame, text="停止", command=self.stop_training, state='disabled'
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.reset_button = ttk.Button(
            control_frame, text="重置", command=self.reset_environment
        )
        self.reset_button.grid(row=0, column=2, padx=5)
        
        # 参数调节
        ttk.Label(control_frame, text="学习率:").grid(row=0, column=3, padx=(20, 5))
        self.lr_var = tk.DoubleVar(value=0.001)
        self.lr_scale = ttk.Scale(
            control_frame, from_=0.0001, to=0.01, 
            variable=self.lr_var, orient=tk.HORIZONTAL, length=150
        )
        self.lr_scale.grid(row=0, column=4, padx=5)
        self.lr_label = ttk.Label(control_frame, text="0.001")
        self.lr_label.grid(row=0, column=5, padx=5)
        
        ttk.Label(control_frame, text="探索率:").grid(row=0, column=6, padx=(20, 5))
        self.epsilon_var = tk.DoubleVar(value=0.3)
        self.epsilon_scale = ttk.Scale(
            control_frame, from_=0.0, to=1.0, 
            variable=self.epsilon_var, orient=tk.HORIZONTAL, length=150
        )
        self.epsilon_scale.grid(row=0, column=7, padx=5)
        self.epsilon_label = ttk.Label(control_frame, text="0.30")
        self.epsilon_label.grid(row=0, column=8, padx=5)
        
        # 更新标签
        self.lr_scale.config(command=lambda v: self.lr_label.config(text=f"{float(v):.4f}"))
        self.epsilon_scale.config(command=lambda v: self.epsilon_label.config(text=f"{float(v):.2f}"))
        
        # 状态显示面板
        status_frame = ttk.LabelFrame(self.root, text="系统状态", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建状态标签
        self.status_labels = {}
        status_items = [
            ("Episode", "0"),
            ("EP Distance", "N/A"),
            ("Sensitivity", "1.00x"),
            ("Total Reward", "0.00"),
            ("Best EP Distance", "N/A"),
            ("Current ε", "0.30")
        ]
        
        for i, (name, initial) in enumerate(status_items):
            ttk.Label(status_frame, text=f"{name}:", font=('Arial', 10, 'bold')).grid(
                row=i // 3, column=(i % 3) * 2, sticky=tk.W, padx=10, pady=3
            )
            label = ttk.Label(status_frame, text=initial, font=('Arial', 10))
            label.grid(row=i // 3, column=(i % 3) * 2 + 1, sticky=tk.W, padx=10, pady=3)
            self.status_labels[name] = label
        
        # 参数显示面板
        param_frame = ttk.LabelFrame(self.root, text="当前参数", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.param_labels = {}
        param_items = [
            ("γ (Loss)", "10.00"),
            ("Pump Power", "50.00"),
            ("δ (Detuning)", "0.00"),
            ("κ (Coupling)", "0.50")
        ]
        
        for i, (name, initial) in enumerate(param_items):
            ttk.Label(param_frame, text=f"{name}:", font=('Arial', 10, 'bold')).grid(
                row=0, column=i*2, sticky=tk.W, padx=10, pady=3
            )
            label = ttk.Label(param_frame, text=initial, font=('Arial', 10))
            label.grid(row=0, column=i*2+1, sticky=tk.W, padx=10, pady=3)
            self.param_labels[name] = label
    
    def _create_plots(self):
        """创建图表"""
        # 创建matplotlib图表
        self.fig = Figure(figsize=(14, 6), dpi=100)
        
        # 子图布局
        self.ax1 = self.fig.add_subplot(2, 3, 1)
        self.ax2 = self.fig.add_subplot(2, 3, 2)
        self.ax3 = self.fig.add_subplot(2, 3, 3)
        self.ax4 = self.fig.add_subplot(2, 3, 4)
        self.ax5 = self.fig.add_subplot(2, 3, 5)
        self.ax6 = self.fig.add_subplot(2, 3, 6)
        
        # 初始化线条
        self.line1, = self.ax1.plot([], [], 'b-', alpha=0.7)
        self.line2, = self.ax2.plot([], [], 'r-', alpha=0.7)
        self.line3, = self.ax3.plot([], [], 'g-', alpha=0.7)
        self.line4, = self.ax4.plot([], [], 'm-', alpha=0.7)
        self.line5, = self.ax5.plot([], [], 'c-', alpha=0.7)
        self.line6, = self.ax6.plot([], [], 'y-', alpha=0.7)
        
        # 设置标题和标签
        self.ax1.set_title('EP Distance Evolution')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('EP Distance')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Reward History')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Reward')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Sensitivity Enhancement')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Enhancement Factor')
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title('Loss Parameter γ')
        self.ax4.set_xlabel('Episode')
        self.ax4.set_ylabel('γ')
        self.ax4.grid(True, alpha=0.3)
        
        self.ax5.set_title('Pump Power')
        self.ax5.set_xlabel('Episode')
        self.ax5.set_ylabel('Power')
        self.ax5.grid(True, alpha=0.3)
        
        self.ax6.set_title('Detuning δ')
        self.ax6.set_xlabel('Episode')
        self.ax6.set_ylabel('δ')
        self.ax6.grid(True, alpha=0.3)
        
        # EP阈值线
        self.ax1.axhline(y=0.1, color='g', linestyle='--', alpha=0.5)
        
        # 嵌入到tkinter
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig.tight_layout()
    
    def update_plots(self):
        """更新图表"""
        with self.lock:
            if len(self.episode_data) > 0:
                # 更新线条数据
                self.line1.set_data(self.episode_data, self.ep_distance_data)
                self.line2.set_data(self.episode_data, self.reward_data)
                self.line3.set_data(self.episode_data, self.sensitivity_data)
                self.line4.set_data(self.episode_data, self.gamma_data)
                self.line5.set_data(self.episode_data, self.pump_data)
                self.line6.set_data(self.episode_data, self.detuning_data)
                
                # 自动调整坐标轴范围
                for ax, data in [
                    (self.ax1, self.ep_distance_data),
                    (self.ax2, self.reward_data),
                    (self.ax3, self.sensitivity_data),
                    (self.ax4, self.gamma_data),
                    (self.ax5, self.pump_data),
                    (self.ax6, self.detuning_data)
                ]:
                    if data:
                        ax.relim()
                        ax.autoscale_view()
                
                # 重绘
                self.canvas.draw()
        
        # 继续更新
        self.root.after(self.update_interval, self.update_plots)
    
    def training_loop(self):
        """训练循环（在后台线程运行）"""
        while self.is_running:
            # 执行一步训练
            info = self.env.step()
            
            # 更新代理参数
            self.env.agent.lr = self.lr_var.get()
            self.env.agent.epsilon = self.epsilon_var.get()
            
            # 记录数据
            with self.lock:
                self.episode_data.append(info['episode'])
                self.ep_distance_data.append(info['ep_distance'])
                self.reward_data.append(info['reward'])
                self.sensitivity_data.append(info['sensitivity'])
                self.gamma_data.append(self.env.params.gamma)
                self.pump_data.append(self.env.params.pump_power)
                self.detuning_data.append(self.env.params.detuning)
                
                # 限制数据长度
                if len(self.episode_data) > self.max_points:
                    self.episode_data.pop(0)
                    self.ep_distance_data.pop(0)
                    self.reward_data.pop(0)
                    self.sensitivity_data.pop(0)
                    self.gamma_data.pop(0)
                    self.pump_data.pop(0)
                    self.detuning_data.pop(0)
            
            # 更新状态标签
            self.update_status_labels(info)
            
            # 控制训练速度
            time.sleep(0.01)
    
    def update_status_labels(self, info):
        """更新状态标签"""
        def update():
            self.status_labels["Episode"].config(text=str(info['episode']))
            self.status_labels["EP Distance"].config(text=f"{info['ep_distance']:.6f}")
            self.status_labels["Sensitivity"].config(text=f"{info['sensitivity']:.2f}x")
            self.status_labels["Total Reward"].config(text=f"{info['total_reward']:.2f}")
            self.status_labels["Best EP Distance"].config(text=f"{self.env.best_ep_distance:.6f}")
            self.status_labels["Current ε"].config(text=f"{info['epsilon']:.3f}")
            
            self.param_labels["γ (Loss)"].config(text=f"{self.env.params.gamma:.2f}")
            self.param_labels["Pump Power"].config(text=f"{self.env.params.pump_power:.2f}")
            self.param_labels["δ (Detuning)"].config(text=f"{self.env.params.detuning:.2f}")
            self.param_labels["κ (Coupling)"].config(text=f"{self.env.params.coupling:.2f}")
            
            # 如果达到EP点，显示提示
            if info['ep_distance'] < 0.1:
                self.status_labels["EP Distance"].config(foreground='green')
            else:
                self.status_labels["EP Distance"].config(foreground='black')
        
        self.root.after(0, update)
    
    def start_training(self):
        """开始训练"""
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.reset_button.config(state='disabled')
            
            # 启动训练线程
            self.training_thread = Thread(target=self.training_loop, daemon=True)
            self.training_thread.start()
    
    def stop_training(self):
        """停止训练"""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.reset_button.config(state='normal')
        
        # 显示最终结果
        if self.env.best_params:
            msg = f"训练完成！\n\n"
            msg += f"最佳EP距离: {self.env.best_ep_distance:.6f}\n"
            msg += f"最优参数:\n"
            msg += f"  γ = {self.env.best_params.gamma:.2f}\n"
            msg += f"  Pump = {self.env.best_params.pump_power:.2f}\n"
            msg += f"  δ = {self.env.best_params.detuning:.2f}\n"
            msg += f"  κ = {self.env.best_params.coupling:.2f}\n"
            
            final_system = NonHermitianSystem(self.env.best_params)
            final_sensitivity = final_system.get_sensitivity_enhancement()
            msg += f"\n灵敏度增强: {final_sensitivity:.2f}x"
            
            messagebox.showinfo("训练完成", msg)
    
    def reset_environment(self):
        """重置环境"""
        self.env.reset()
        
        # 清空数据
        with self.lock:
            self.episode_data.clear()
            self.ep_distance_data.clear()
            self.reward_data.clear()
            self.sensitivity_data.clear()
            self.gamma_data.clear()
            self.pump_data.clear()
            self.detuning_data.clear()
        
        # 重置状态标签
        for label in self.status_labels.values():
            label.config(text="N/A", foreground='black')
        self.status_labels["Episode"].config(text="0")
        self.status_labels["Total Reward"].config(text="0.00")
        self.status_labels["Current ε"].config(text=f"{self.epsilon_var.get():.2f}")
        
        # 重置参数标签
        self.param_labels["γ (Loss)"].config(text="10.00")
        self.param_labels["Pump Power"].config(text="50.00")
        self.param_labels["δ (Detuning)"].config(text="0.00")
        self.param_labels["κ (Coupling)"].config(text="0.50")

def main():
    """主函数"""
    root = tk.Tk()
    app = RealTimeDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
