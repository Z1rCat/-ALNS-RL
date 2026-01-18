#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALNS Research Visualization Code
高级科研风格图表生成代码
作者: AI研究团队
项目: 34959_RL - ALNS多式联运动态优化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrow
import seaborn as sns
from matplotlib import gridspec
import networkx as nx
from matplotlib.patches import ConnectionPatch
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和全局样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 科研风格颜色配置
COLORS = {
    'primary': '#2E86AB',      # 深蓝色 - 主色调
    'secondary': '#A23B72',    # 紫红色 - 次要色
    'accent1': '#F18F01',      # 橙色 - 强调色1
    'accent2': '#C73E1D',      # 红色 - 强调色2
    'accent3': '#4CAF50',      # 绿色 - 强调色3
    'dark': '#1a1a1a',         # 深灰色
    'light': '#f0f0f0',        # 浅灰色
    'gradient_start': '#E8F4FD', # 渐变起始色
    'gradient_end': '#2E86AB'     # 渐变结束色
}

def set_scientific_style():
    """设置科研论文风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

def create_alns_framework():
    """创建ALNS算法框架图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, 'ALNS Algorithm Framework',
            fontsize=18, fontweight='bold', ha='center',
            color=COLORS['dark'], family='Arial')

    # 主要组件 - 使用圆角矩形
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
                color='white', fontweight='bold', family='Arial')

    # 箭头连接
    arrows = [
        (5, 8.1, 5, 7.4),    # Initial to Removal
        (5, 8.1, 5, 7.4),    # Initial to Insertion
        (2, 6.6, 3.7, 5.9),  # Removal to Selection
        (8, 6.6, 6.3, 5.9),  # Insertion to Selection
        (5, 5.1, 5, 4.4),    # Selection to Local Search
        (5, 3.6, 5, 2.9),    # Local Search to Acceptance
        (5, 2.1, 5, 1.4),    # Acceptance to Best
    ]

    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrow(x1, y1, x2-x1, y2-y1,
                          width=0.05, head_width=0.15, head_length=0.1,
                          fc=COLORS['dark'], ec=COLORS['dark'], alpha=0.7)
        ax.add_patch(arrow)

    # 添加注释
    ax.text(0.5, 8.5, 'Generate\ninitial\nfeasible solution',
            fontsize=9, ha='center', va='center', style='italic')
    ax.text(0.5, 7, 'Remove q\nrequests\ncurrent solution',
            fontsize=9, ha='center', va='center', style='italic')
    ax.text(9.5, 7, 'Reinsert removed\nrequests using\nheuristic methods',
            fontsize=9, ha='center', va='center', style='italic')
    ax.text(0.5, 5.5, 'Adaptive\noperator\nselection',
            fontsize=9, ha='center', va='center', style='italic')

    plt.savefig('figures/alns_framework.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ ALNS框架图已生成")

def create_system_architecture():
    """创建系统架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(6, 9.5, 'Multi-modal Transportation ALNS System Architecture',
            fontsize=18, fontweight='bold', ha='center',
            color=COLORS['dark'], family='Arial')

    # 主要模块
    modules = [
        # 输入层
        (2, 8.5, 2.5, 1, 'Network\nData', COLORS['gradient_start']),
        (6, 8.5, 2.5, 1, 'Uncertainty\nEvents', COLORS['gradient_start']),
        (10, 8.5, 2.5, 1, 'Optimization\nParameters', COLORS['gradient_start']),

        # 核心处理层
        (1.5, 6.5, 2, 1.2, 'Initial\nSolution\nGenerator', COLORS['primary']),
        (4.5, 6.5, 3, 1.2, 'ALNS\nCore\nEngine', COLORS['secondary']),
        (8.5, 6.5, 2, 1.2, 'RL\nDecision\nModule', COLORS['accent1']),

        # 输出层
        (3, 4.5, 2.5, 1, 'Optimized\nRoutes', COLORS['accent2']),
        (7, 4.5, 2.5, 1, 'Performance\nMetrics', COLORS['accent2']),

        # 底层支撑
        (1.5, 2.5, 2.5, 0.8, 'Removal\nOperators', COLORS['accent3']),
        (4.5, 2.5, 2.5, 0.8, 'Insertion\nOperators', COLORS['accent3']),
        (7.5, 2.5, 2.5, 0.8, 'Adaptive\nMechanism', COLORS['accent3']),
    ]

    for x, y, w, h, label, color in modules:
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white',
                              linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=10, ha='center', va='center',
                color='white', fontweight='bold', family='Arial')

    # 连接线
    connections = [
        # 输入到核心
        (2, 8, 2.5, 7.1), (6, 8, 6, 7.1), (10, 8, 9.5, 7.1),
        # 核心之间
        (2.5, 6.5, 4.5, 6.5), (6, 6.5, 8.5, 6.5),
        # 核心到输出
        (4.5, 5.9, 3, 5), (6, 5.9, 7, 5),
        # 支撑到核心
        (1.5, 2.9, 4.5, 5.9), (4.5, 2.9, 6, 5.9), (7.5, 2.9, 7.5, 5.9),
    ]

    for x1, y1, x2, y2 in connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1.5,
                                 color=COLORS['dark'], alpha=0.6))

    # 添加数据流标签
    ax.text(4, 8.2, 'Input', fontsize=9, ha='center', style='italic')
    ax.text(6, 5.5, 'Processing', fontsize=9, ha='center', style='italic')
    ax.text(6, 4, 'Output', fontsize=9, ha='center', style='italic')
    ax.text(6, 2, 'Support Components', fontsize=9, ha='center', style='italic')

    plt.savefig('figures/system_architecture.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ 系统架构图已生成")

def create_performance_comparison():
    """创建算法性能对比图"""
    set_scientific_style()

    # 模拟数据
    algorithms = ['ALNS-RL', 'Standard ALNS', 'Genetic Algorithm', 'Tabu Search', 'Simulated Annealing']
    instances = ['Small (5-20)', 'Medium (50-100)', 'Large (200-700)']

    # 生成模拟性能数据
    np.random.seed(42)
    performance_data = {
        'Cost': np.array([
            [100, 95, 88],  # ALNS-RL
            [108, 102, 95], # Standard ALNS
            [115, 112, 105], # GA
            [118, 115, 110], # TS
            [120, 118, 115]  # SA
        ]),
        'Time': np.array([
            [45, 380, 2500],   # ALNS-RL
            [50, 420, 2800],   # Standard ALNS
            [65, 580, 3500],   # GA
            [55, 480, 2900],   # TS
            [70, 620, 3800]    # SA
        ])
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 成本对比
    x = np.arange(len(instances))
    width = 0.15
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent1'],
              COLORS['accent2'], COLORS['accent3']]

    for i, algo in enumerate(algorithms):
        ax1.bar(x + i*width, performance_data['Cost'][i], width,
               label=algo, color=colors[i], alpha=0.8)

    ax1.set_xlabel('Problem Scale')
    ax1.set_ylabel('Total Cost (×1000 €)')
    ax1.set_title('Cost Comparison Across Different Problem Scales')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(instances)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 时间对比
    for i, algo in enumerate(algorithms):
        ax2.bar(x + i*width, performance_data['Time'][i], width,
               label=algo, color=colors[i], alpha=0.8)

    ax2.set_xlabel('Problem Scale')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('Computation Time Comparison')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(instances)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # 使用对数坐标
    ax2.set_yscale('log')
    ax2.set_ylabel('Computation Time (seconds, log scale)')

    plt.tight_layout()
    plt.savefig('figures/performance_comparison.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ 性能对比图已生成")

def create_operator_weights_evolution():
    """创建算子权重演化图"""
    set_scientific_style()

    # 模拟权重演化数据
    iterations = np.arange(0, 1000)
    n_operators = 4

    # 生成权重演化轨迹
    weights = np.zeros((n_operators, len(iterations)))
    weights[:, 0] = 0.25  # 初始等权重

    for i in range(1, len(iterations)):
        # 模拟权重的自适应变化
        for j in range(n_operators):
            change = np.random.normal(0, 0.01)
            weights[j, i] = max(0.05, weights[j, i-1] + change)

        # 归一化
        weights[:, i] = weights[:, i] / np.sum(weights[:, i])

    operator_names = ['Random Removal', 'Worst Removal', 'Related Removal', 'History Removal']
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent1'], COLORS['accent2']]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i in range(n_operators):
        ax.plot(iterations, weights[i], linewidth=2.5,
               label=operator_names[i], color=colors[i])

    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Operator Weight')
    ax.set_title('Adaptive Weight Evolution of Removal Operators')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(weights.flatten()) * 1.1)

    # 添加收敛区域标注
    convergence_start = 600
    ax.axvspan(convergence_start, iterations[-1], alpha=0.1, color='green')
    ax.text(iterations[-1] * 0.8, max(weights.flatten()) * 0.9,
           'Convergence Region', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig('figures/operator_weights_evolution.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ 算子权重演化图已生成")

def create_transportation_network():
    """创建多式联运网络图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # 创建网络图
    G = nx.Graph()

    # 添加节点 (起点、终点、中转站)
    nodes = {
        'Start_A': (2, 8),
        'Start_B': (2, 6),
        'Terminal_1': (5, 8),
        'Terminal_2': (5, 6),
        'Terminal_3': (8, 7),
        'End_A': (11, 8),
        'End_B': (11, 6)
    }

    for node, pos in nodes.items():
        G.add_node(node, pos=pos)

    # 添加边 (运输路线)
    edges = [
        ('Start_A', 'Terminal_1', 'truck'),
        ('Start_B', 'Terminal_2', 'truck'),
        ('Terminal_1', 'Terminal_3', 'barge'),
        ('Terminal_2', 'Terminal_3', 'train'),
        ('Terminal_3', 'End_A', 'truck'),
        ('Terminal_3', 'End_B', 'truck'),
        ('Terminal_1', 'Terminal_2', 'truck')
    ]

    for u, v, mode in edges:
        G.add_edge(u, v, mode=mode)

    # 绘制节点
    pos = nx.get_node_attributes(G, 'pos')

    # 区分不同类型的节点
    start_nodes = ['Start_A', 'Start_B']
    terminal_nodes = ['Terminal_1', 'Terminal_2', 'Terminal_3']
    end_nodes = ['End_A', 'End_B']

    # 绘制起点
    nx.draw_networkx_nodes(G, pos, nodelist=start_nodes,
                          node_color=COLORS['accent3'], node_size=800,
                          ax=ax)

    # 绘制中转站
    nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes,
                          node_color=COLORS['primary'], node_size=1000,
                          ax=ax)

    # 绘制终点
    nx.draw_networkx_nodes(G, pos, nodelist=end_nodes,
                          node_color=COLORS['accent2'], node_size=800,
                          ax=ax)

    # 绘制边（区分运输方式）
    truck_edges = [(u, v) for u, v, mode in edges if mode == 'truck']
    barge_edges = [(u, v) for u, v, mode in edges if mode == 'barge']
    train_edges = [(u, v) for u, v, mode in edges if mode == 'train']

    nx.draw_networkx_edges(G, pos, edgelist=truck_edges,
                          edge_color=COLORS['dark'], width=2, style='-', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=barge_edges,
                          edge_color=COLORS['secondary'], width=3, style='--', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=train_edges,
                          edge_color=COLORS['accent1'], width=3, style='-.', ax=ax)

    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['dark'], lw=2, label='Truck'),
        plt.Line2D([0], [0], color=COLORS['secondary'], lw=3, linestyle='--', label='Barge'),
        plt.Line2D([0], [0], color=COLORS['accent1'], lw=3, linestyle='-.', label='Train'),
        plt.scatter([], [], c=COLORS['accent3'], s=100, label='Start Points'),
        plt.scatter([], [], c=COLORS['primary'], s=150, label='Transfer Terminals'),
        plt.scatter([], [], c=COLORS['accent2'], s=100, label='End Points')
    ]

    ax.legend(handles=legend_elements, loc='upper right',
             bbox_to_anchor=(1.15, 1), frameon=True, fancybox=True)

    ax.set_title('Multi-modal Transportation Network Example',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/transportation_network.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ 多式联运网络图已生成")

def create_rl_training_curves():
    """创建强化学习训练曲线"""
    set_scientific_style()

    # 模拟训练数据
    episodes = np.arange(0, 1000)

    # 生成不同算法的奖励曲线
    def generate_reward_curve(episodes, noise_level=0.1, trend=0.001):
        rewards = []
        current_reward = 0.1
        for i in episodes:
            noise = np.random.normal(0, noise_level)
            improvement = trend * (1 - np.exp(-i/100))
            current_reward = max(0, min(1, current_reward + improvement + noise))
            rewards.append(current_reward)
        return np.array(rewards)

    # DQN
    dqn_rewards = generate_reward_curve(episodes, 0.15, 0.002)
    # PPO
    ppo_rewards = generate_reward_curve(episodes, 0.12, 0.0018)
    # A2C
    a2c_rewards = generate_reward_curve(episodes, 0.18, 0.0015)

    # 添加滑动平均
    window_size = 50
    dqn_smooth = np.convolve(dqn_rewards, np.ones(window_size)/window_size, mode='valid')
    ppo_smooth = np.convolve(ppo_rewards, np.ones(window_size)/window_size, mode='valid')
    a2c_smooth = np.convolve(a2c_rewards, np.ones(window_size)/window_size, mode='valid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

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
    ax1.set_ylim(0, 1.0)

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
    print("✓ 强化学习训练曲线已生成")

def create_uncertainty_handling():
    """创建不确定性处理流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(5, 9.5, 'Dynamic Uncertainty Event Handling',
            fontsize=16, fontweight='bold', ha='center',
            color=COLORS['dark'], family='Arial')

    # 事件检测
    event_detect = FancyBboxPatch((1, 8), 3, 0.8,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLORS['accent1'],
                                  edgecolor='white', linewidth=2)
    ax.add_patch(event_detect)
    ax.text(2.5, 8.4, 'Event Detection', fontsize=11, ha='center',
            va='center', color='white', fontweight='bold')

    # 影响评估
    impact_assess = FancyBboxPatch((6, 8), 3, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=COLORS['secondary'],
                                   edgecolor='white', linewidth=2)
    ax.add_patch(impact_assess)
    ax.text(7.5, 8.4, 'Impact Assessment', fontsize=11, ha='center',
            va='center', color='white', fontweight='bold')

    # RL决策
    rl_decision = FancyBboxPatch((3.5, 6), 3, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor=COLORS['primary'],
                                edgecolor='white', linewidth=2)
    ax.add_patch(rl_decision)
    ax.text(5, 6.4, 'RL Decision Making', fontsize=11, ha='center',
            va='center', color='white', fontweight='bold')

    # 执行策略
    strategies = [
        (1, 4, 2, 0.6, 'Wait & Monitor', COLORS['accent3']),
        (4, 4, 2, 0.6, 'Local Rerouting', COLORS['accent3']),
        (7, 4, 2, 0.6, 'Global Replan', COLORS['accent3'])
    ]

    for x, y, w, h, label, color in strategies:
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=10, ha='center', va='center',
                color='white', fontweight='bold')

    # 结果评估
    result_eval = FancyBboxPatch((5, 2), 3, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor=COLORS['accent2'],
                                edgecolor='white', linewidth=2)
    ax.add_patch(result_eval)
    ax.text(6.5, 2.4, 'Result Evaluation', fontsize=11, ha='center',
            va='center', color='white', fontweight='bold')

    # 连接箭头
    arrows = [
        (4, 8.4, 6, 8.4),    # 检测到评估
        (7.5, 8, 5, 6.8),    # 评估到决策
        (5, 5.6, 5, 4.3),    # 决策到策略
        (1, 3.7, 4, 2.8),    # 策略到评估
        (4, 3.7, 4, 2.8),
        (7, 3.7, 6.5, 2.8),
    ]

    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrow(x1, y1, x2-x1, y2-y1,
                          width=0.03, head_width=0.12, head_length=0.08,
                          fc=COLORS['dark'], ec=COLORS['dark'], alpha=0.7)
        ax.add_patch(arrow)

    # 添加时间轴
    timeline_x = [0.5, 2, 3.5, 5, 6.5, 8, 9.5]
    timeline_y = [0.5] * len(timeline_x)
    ax.plot(timeline_x, timeline_y, 'o-', color=COLORS['dark'],
            linewidth=2, markersize=8)

    timeline_labels = ['t₀', 't₁', 't₂', 't₃', 't₄', 't₅', 't₆']
    for x, label in zip(timeline_x, timeline_labels):
        ax.text(x, 0.2, label, fontsize=10, ha='center',
                fontweight='bold', color=COLORS['dark'])

    ax.text(5, 0.8, 'Timeline', fontsize=11, ha='center',
            style='italic', color=COLORS['dark'])

    plt.savefig('figures/uncertainty_handling.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ 不确定性处理流程图已生成")

def create_multi_objective_pareto():
    """创建多目标帕累托前沿图"""
    set_scientific_style()

    # 生成帕累托前沿数据
    np.random.seed(42)
    n_solutions = 200

    # 生成一些随机解
    cost = np.random.uniform(80, 150, n_solutions)
    time = np.random.uniform(40, 100, n_solutions)
    emission = np.random.uniform(20, 60, n_solutions)

    # 创建帕累托最优解
    pareto_cost = np.linspace(85, 120, 50)
    pareto_time = 120 - 0.5 * pareto_cost + np.random.normal(0, 2, 50)
    pareto_emission = 0.3 * pareto_cost + 10 + np.random.normal(0, 1, 50)

    fig = plt.figure(figsize=(16, 12))

    # 3D散点图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(cost, time, emission, alpha=0.4, c='gray', s=30, label='Dominated Solutions')
    ax1.scatter(pareto_cost, pareto_time, pareto_emission,
               c=COLORS['primary'], s=50, label='Pareto Optimal')

    ax1.set_xlabel('Total Cost (×1000 €)')
    ax1.set_ylabel('Total Time (h)')
    ax1.set_zlabel('Emissions (tons)')
    ax1.set_title('3D Pareto Front')
    ax1.legend()

    # 2D投影：成本 vs 时间
    ax2 = fig.add_subplot(222)
    ax2.scatter(cost, time, alpha=0.4, c='gray', s=20, label='Dominated')
    ax2.scatter(pareto_cost, pareto_time,
               c=COLORS['primary'], s=40, label='Pareto Optimal')
    ax2.set_xlabel('Total Cost (×1000 €)')
    ax2.set_ylabel('Total Time (h)')
    ax2.set_title('Cost vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 2D投影：成本 vs 排放
    ax3 = fig.add_subplot(223)
    ax3.scatter(cost, emission, alpha=0.4, c='gray', s=20, label='Dominated')
    ax3.scatter(pareto_cost, pareto_emission,
               c=COLORS['secondary'], s=40, label='Pareto Optimal')
    ax3.set_xlabel('Total Cost (×1000 €)')
    ax3.set_ylabel('Emissions (tons)')
    ax3.set_title('Cost vs Emissions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 2D投影：时间 vs 排放
    ax4 = fig.add_subplot(224)
    ax4.scatter(time, emission, alpha=0.4, c='gray', s=20, label='Dominated')
    ax4.scatter(pareto_time, pareto_emission,
               c=COLORS['accent1'], s=40, label='Pareto Optimal')
    ax4.set_xlabel('Total Time (h)')
    ax4.set_ylabel('Emissions (tons)')
    ax4.set_title('Time vs Emissions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/multi_objective_pareto.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ 多目标帕累托前沿图已生成")

def main():
    """主函数：生成所有图表"""
    print("🎨 开始生成ALNS研究论文图表...")
    print("=" * 50)

    create_alns_framework()
    create_system_architecture()
    create_performance_comparison()
    create_operator_weights_evolution()
    create_transportation_network()
    create_rl_training_curves()
    create_uncertainty_handling()
    create_multi_objective_pareto()

    print("=" * 50)
    print("✅ 所有图表生成完成！")
    print("📁 图表已保存到 'figures/' 文件夹")
    print("\n生成的图表包括：")
    print("  1. ALNS算法框架图")
    print("  2. 系统架构图")
    print("  3. 算法性能对比图")
    print("  4. 算子权重演化图")
    print("  5. 多式联运网络图")
    print("  6. 强化学习训练曲线")
    print("  7. 不确定性处理流程图")
    print("  8. 多目标帕累托前沿图")

if __name__ == "__main__":
    main()