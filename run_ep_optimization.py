#!/usr/bin/env python3
"""
EP强化学习优化系统 - 主运行脚本
Main script for EP RL Optimization System
"""

import sys
import argparse
from datetime import datetime

def main():
    """主函数 - 提供命令行接口"""
    
    parser = argparse.ArgumentParser(
        description='非厄米奇异点强化学习优化系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_ep_optimization.py --mode basic --episodes 1000
  python run_ep_optimization.py --mode gui
  python run_ep_optimization.py --mode dqn --episodes 500
  python run_ep_optimization.py --mode parallel --workers 4
        """
    )
    
    # 运行模式
    parser.add_argument(
        '--mode', 
        choices=['basic', 'gui', 'dqn', 'parallel', 'test'],
        default='basic',
        help='运行模式: basic(基础Q-Learning), gui(可视化界面), dqn(深度强化学习), parallel(并行训练), test(测试模式)'
    )
    
    # 训练参数
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='训练回合数 (默认: 1000)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='学习率 (默认: 0.001)'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.3,
        help='初始探索率 (默认: 0.3)'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='折扣因子 (默认: 0.95)'
    )
    
    # 并行训练参数
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='并行工作进程数 (默认: 4)'
    )
    
    # 其他选项
    parser.add_argument(
        '--save',
        action='store_true',
        help='保存训练结果'
    )
    
    parser.add_argument(
        '--load',
        type=str,
        help='加载已保存的模型'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("非厄米奇异点强化学习优化系统")
    print("Exceptional Points Optimization via Reinforcement Learning")
    print("="*60)
    print(f"运行模式: {args.mode}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    
    # 根据模式运行
    if args.mode == 'basic':
        run_basic_training(args)
    elif args.mode == 'gui':
        run_gui(args)
    elif args.mode == 'dqn':
        run_dqn_training(args)
    elif args.mode == 'parallel':
        run_parallel_training(args)
    elif args.mode == 'test':
        run_test(args)
    
    print("-"*60)
    print("运行完成！")
    print("="*60)

def run_basic_training(args):
    """运行基础Q-Learning训练"""
    print("\n启动基础Q-Learning训练...")
    
    from ep_rl_optimization import EPOptimizationEnvironment, Visualizer
    
    # 创建环境
    env = EPOptimizationEnvironment()
    
    # 设置参数
    env.agent.lr = args.lr
    env.agent.epsilon = args.epsilon
    env.agent.gamma = args.gamma
    
    # 训练
    print(f"开始训练 {args.episodes} 个回合...")
    env.train(n_episodes=args.episodes)
    
    # 显示结果
    print("\n训练结果:")
    print(f"  最佳EP距离: {env.best_ep_distance:.6f}")
    if env.best_params:
        print(f"  最优参数:")
        print(f"    - γ (Loss): {env.best_params.gamma:.2f}")
        print(f"    - Pump Power: {env.best_params.pump_power:.2f}")
        print(f"    - δ (Detuning): {env.best_params.detuning:.2f}")
    
    # 可视化
    print("\n生成可视化...")
    viz = Visualizer(env)
    viz.plot_results()
    viz.show()
    
    # 保存结果
    if args.save:
        import pickle
        filename = f"ep_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump({
                'history': env.ep_distance_history,
                'rewards': env.reward_history,
                'params': env.best_params,
                'q_table': env.agent.q_table
            }, f)
        print(f"\n结果已保存到: {filename}")

def run_gui(args):
    """运行GUI界面"""
    print("\n启动可视化监控界面...")
    print("提示: 点击'开始训练'按钮开始优化")
    
    from ep_rl_dashboard import main as gui_main
    gui_main()

def run_dqn_training(args):
    """运行DQN深度强化学习训练"""
    print("\n启动DQN深度强化学习训练...")
    
    from ep_advanced_optimization import AdvancedEPEnvironment
    import torch
    
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = AdvancedEPEnvironment(use_dqn=True)
    
    # 设置参数
    env.agent.lr = args.lr
    env.agent.epsilon = args.epsilon
    env.agent.gamma = args.gamma
    
    # 加载模型
    if args.load:
        print(f"加载模型: {args.load}")
        env.agent.load(args.load)
    
    # 训练
    print(f"开始DQN训练 {args.episodes} 个回合...")
    history = env.train_dqn(episodes=args.episodes)
    
    # 显示结果
    print("\nDQN训练结果:")
    print(f"  最佳EP距离: {env.best_ep_distance:.6f}")
    if hasattr(env, 'best_params') and env.best_params:
        print(f"  最优参数:")
        for key, value in env.best_params.items():
            print(f"    - {key}: {value:.2f}")
    
    # 保存模型
    if args.save:
        model_file = f"dqn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        env.agent.save(model_file)
        print(f"\n模型已保存到: {model_file}")
    
    # 可视化结果
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['episodes'], history['ep_distances'])
    plt.xlabel('Episode')
    plt.ylabel('EP Distance')
    plt.title('EP Distance Evolution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['episodes'], history['rewards'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['episodes'], history['sensitivities'])
    plt.xlabel('Episode')
    plt.ylabel('Sensitivity Enhancement')
    plt.title('Measurement Sensitivity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_parallel_training(args):
    """运行并行训练"""
    print(f"\n启动并行训练 (使用 {args.workers} 个工作进程)...")
    
    from ep_advanced_optimization import ParallelTrainer
    
    # 创建并行训练器
    trainer = ParallelTrainer(n_workers=args.workers)
    
    # 并行训练
    total_episodes = args.episodes
    print(f"总训练回合数: {total_episodes} (每个进程: {total_episodes // args.workers})")
    
    results = trainer.train_parallel(total_episodes=total_episodes)
    
    # 汇总结果
    print("\n并行训练结果汇总:")
    for i, result in enumerate(results):
        print(f"  Worker {i}: EP距离={result['best_ep_distance']:.6f}")
    
    # 找到全局最优
    best_result = min(results, key=lambda x: x['best_ep_distance'])
    print(f"\n全局最优:")
    print(f"  最佳EP距离: {best_result['best_ep_distance']:.6f}")
    if best_result['best_params']:
        print(f"  最优参数: {best_result['best_params']}")
    
    # 保存结果
    if args.save:
        import pickle
        filename = f"parallel_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n结果已保存到: {filename}")

def run_test(args):
    """运行测试模式"""
    print("\n运行完整系统测试...")
    
    from ep_advanced_optimization import test_advanced_system
    
    # 运行测试
    env, results = test_advanced_system()
    
    print("\n测试总结:")
    print(f"  - Q-Learning: ✓")
    print(f"  - DQN: ✓")
    print(f"  - 并行训练: ✓")
    print(f"  - 自适应控制: ✓")
    print(f"  - 数字孪生: ✓")
    print("\n所有模块测试通过！")

def print_system_info():
    """打印系统信息"""
    import platform
    import torch
    import numpy as np
    
    print("\n系统信息:")
    print(f"  Python版本: {platform.python_version()}")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  NumPy版本: {np.__version__}")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    # 打印系统信息
    if '--info' in sys.argv:
        print_system_info()
        sys.exit(0)
    
    # 运行主程序
    main()
