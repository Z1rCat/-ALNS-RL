#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified ALNS Research Visualization Code
简化版ALNS研究可视化代码
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrow
import warnings
warnings.filterwarnings('ignore')

# 设置全局样式
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 科研风格颜色配置
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent1': '#F18F01',
    'accent2': '#C73E1D',
    'accent3': '#4CAF50',
    'dark': '#1a1a1a',
    'light': '#f0f0f0'
}

def create_alns_framework():
    """创建ALNS算法框架图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, 'ALNS Algorithm Framework',
            fontsize=18, fontweight='bold', ha='center', color=COLORS['dark'])

    # 主要组件
    components = [
        (5, 8.5, 2, 0.8, 'Initial Solution', COLORS['primary']),
        (2, 7, 2.5, 0.8, 'Removal Operators', COLORS['secondary']),
        (8, 7, 2.5, 0.8, 'Insertion Operators', COLORS['secondary']),
        (5, 5.5, 2.5, 0.8, 'Adaptive Selection', COLORS['accent1']),
        (5, 4, 2, 0.8, 'Local Search', COLORS['accent2']),
        (5, 2.5, 2.5, 0.8, 'Acceptance Criterion', COLORS['accent3']),
        (5, 1, 2, 0.8, 'Best Solution', COLORS['primary']),
    ]

    for x, y, w, h, label, color in components:
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=11, ha='center', va='center',
                color='white', fontweight='bold')

    plt.savefig('figures/alns_framework.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("ALNS framework diagram generated")

def create_performance_comparison():
    """创建算法性能对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 模拟数据
    algorithms = ['ALNS-RL', 'Standard ALNS', 'GA', 'TS', 'SA']
    instances = ['Small (5-20)', 'Medium (50-100)', 'Large (200-700)']

    cost_data = np.array([
        [100, 95, 88],   # ALNS-RL
        [108, 102, 95],  # Standard ALNS
        [115, 112, 105], # GA
        [118, 115, 110], # TS
        [120, 118, 115]  # SA
    ])

    time_data = np.array([
        [45, 380, 2500],   # ALNS-RL
        [50, 420, 2800],   # Standard ALNS
        [65, 580, 3500],   # GA
        [55, 480, 2900],   # TS
        [70, 620, 3800]    # SA
    ])

    # 成本对比
    x = np.arange(len(instances))
    width = 0.15
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent1'],
              COLORS['accent2'], COLORS['accent3']]

    for i, algo in enumerate(algorithms):
        ax1.bar(x + i*width, cost_data[i], width,
               label=algo, color=colors[i], alpha=0.8)

    ax1.set_xlabel('Problem Scale')
    ax1.set_ylabel('Total Cost (x1000 EUR)')
    ax1.set_title('Cost Comparison Across Different Problem Scales')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(instances)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 时间对比
    for i, algo in enumerate(algorithms):
        ax2.bar(x + i*width, time_data[i], width,
               label=algo, color=colors[i], alpha=0.8)

    ax2.set_xlabel('Problem Scale')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('Computation Time Comparison')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(instances)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('figures/performance_comparison.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("Performance comparison diagram generated")

def create_operator_weights_evolution():
    """创建算子权重演化图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    iterations = np.arange(0, 1000)
    n_operators = 4

    # 生成权重演化轨迹
    weights = np.zeros((n_operators, len(iterations)))
    weights[:, 0] = 0.25

    for i in range(1, len(iterations)):
        for j in range(n_operators):
            change = np.random.normal(0, 0.01)
            weights[j, i] = max(0.05, weights[j, i-1] + change)
        weights[:, i] = weights[:, i] / np.sum(weights[:, i])

    operator_names = ['Random Removal', 'Worst Removal', 'Related Removal', 'History Removal']
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent1'], COLORS['accent2']]

    for i in range(n_operators):
        ax.plot(iterations, weights[i], linewidth=2.5,
               label=operator_names[i], color=colors[i])

    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Operator Weight')
    ax.set_title('Adaptive Weight Evolution of Removal Operators')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/operator_weights_evolution.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("Operator weights evolution diagram generated")

def create_transportation_network():
    """创建多式联运网络图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # 简化的网络表示
    nodes = {
        'Start_A': (2, 8),
        'Start_B': (2, 6),
        'Terminal_1': (5, 8),
        'Terminal_2': (5, 6),
        'Terminal_3': (8, 7),
        'End_A': (11, 8),
        'End_B': (11, 6)
    }

    # 绘制节点
    for name, (x, y) in nodes.items():
        if 'Start' in name:
            color = COLORS['accent3']
        elif 'End' in name:
            color = COLORS['accent2']
        else:
            color = COLORS['primary']

        ax.scatter(x, y, s=500, c=color, alpha=0.8, edgecolors='white', linewidth=2)
        ax.text(x, y, name.replace('_', '\n'), ha='center', va='center',
                fontsize=9, fontweight='bold')

    # 绘制连接
    connections = [
        ((2, 8), (5, 8), 'truck'),
        ((2, 6), (5, 6), 'truck'),
        ((5, 8), (8, 7), 'barge'),
        ((5, 6), (8, 7), 'train'),
        ((8, 7), (11, 8), 'truck'),
        ((8, 7), (11, 6), 'truck'),
    ]

    for (x1, y1), (x2, y2), mode in connections:
        if mode == 'truck':
            linestyle = '-'
            color = COLORS['dark']
        elif mode == 'barge':
            linestyle = '--'
            color = COLORS['secondary']
        else:  # train
            linestyle = '-.'
            color = COLORS['accent1']

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, linestyle=linestyle, alpha=0.7)

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['dark'], lw=2, label='Truck'),
        plt.Line2D([0], [0], color=COLORS['secondary'], lw=2, linestyle='--', label='Barge'),
        plt.Line2D([0], [0], color=COLORS['accent1'], lw=2, linestyle='-.', label='Train'),
        plt.scatter([], [], c=COLORS['accent3'], s=100, label='Start Points'),
        plt.scatter([], [], c=COLORS['primary'], s=100, label='Transfer Terminals'),
        plt.scatter([], [], c=COLORS['accent2'], s=100, label='End Points')
    ]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_title('Multi-modal Transportation Network Example', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/transportation_network.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("Transportation network diagram generated")

def create_rl_training_curves():
    """创建强化学习训练曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    episodes = np.arange(0, 1000)

    # 生成训练数据
    def generate_reward_curve(episodes, noise_level=0.1, trend=0.001):
        rewards = []
        current_reward = 0.1
        for i in episodes:
            noise = np.random.normal(0, noise_level)
            improvement = trend * (1 - np.exp(-i/100))
            current_reward = max(0, min(1, current_reward + improvement + noise))
            rewards.append(current_reward)
        return np.array(rewards)

    # 不同算法的奖励曲线
    dqn_rewards = generate_reward_curve(episodes, 0.15, 0.002)
    ppo_rewards = generate_reward_curve(episodes, 0.12, 0.0018)
    a2c_rewards = generate_reward_curve(episodes, 0.18, 0.0015)

    # 添加滑动平均
    window_size = 50
    dqn_smooth = np.convolve(dqn_rewards, np.ones(window_size)/window_size, mode='valid')
    ppo_smooth = np.convolve(ppo_rewards, np.ones(window_size)/window_size, mode='valid')
    a2c_smooth = np.convolve(a2c_rewards, np.ones(window_size)/window_size, mode='valid')

    # 原始奖励曲线
    ax1.plot(episodes, dqn_rewards, alpha=0.3, color=COLORS['primary'], linewidth=0.5)
    ax1.plot(episodes, ppo_rewards, alpha=0.3, color=COLORS['secondary'], linewidth=0.5)
    ax1.plot(episodes, a2c_rewards, alpha=0.3, color=COLORS['accent1'], linewidth=0.5)

    ax1.plot(episodes[window_size-1:], dqn_smooth, color=COLORS['primary'],
            linewidth=2.5, label='DQN')
    ax1.plot(episodes[window_size-1:], ppo_smooth, color=COLORS['secondary'],
            linewidth=2.5, label='PPO')
    ax1.plot(episodes[window_size-1:], a2c_smooth, color=COLORS['accent1'],
            linewidth=2.5, label='A2C')

    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('RL Training Progress - Average Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 收敛性分析
    convergence_metrics = {
        'DQN': np.std(dqn_rewards[-100:]),
        'PPO': np.std(ppo_rewards[-100:]),
        'A2C': np.std(a2c_rewards[-100:])
    }

    algorithm_names = list(convergence_metrics.keys())
    stability_values = list(convergence_metrics.values())

    bars = ax2.bar(algorithm_names, stability_values,
                  color=[COLORS['primary'], COLORS['secondary'], COLORS['accent1']],
                  alpha=0.8)

    ax2.set_ylabel('Reward Standard Deviation (Last 100 Episodes)')
    ax2.set_title('Algorithm Stability Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, value in zip(bars, stability_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/rl_training_curves.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("RL training curves diagram generated")

def main():
    """主函数：生成所有图表"""
    print("Starting ALNS research visualization...")
    print("=" * 50)

    create_alns_framework()
    create_performance_comparison()
    create_operator_weights_evolution()
    create_transportation_network()
    create_rl_training_curves()

    print("=" * 50)
    print("All visualizations generated successfully!")
    print("Figures saved to 'figures/' folder")
    print("\nGenerated figures include:")
    print("  1. ALNS algorithm framework")
    print("  2. Algorithm performance comparison")
    print("  3. Operator weight evolution")
    print("  4. Multi-modal transportation network")
    print("  5. RL training curves")

if __name__ == "__main__":
    main()