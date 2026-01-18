# Dynamic RL 34959 实现分析报告

## 1. 项目概述

`dynamic_RL34959.py` 是一个用于处理多式联运动态不确定性问题的强化学习系统，主要用于处理运输网络中的拥堵事件。该系统结合了深度Q网络（DQN）和自适应大邻域搜索（ALNS）算法，实现了一个智能决策框架来应对动态拥堵事件。

## 2. 系统架构

### 2.1 核心组件

该系统主要由以下核心组件构成：

- **强化学习环境** (`coordinationEnv`): 自定义的Gym环境
- **DQN智能体**: 使用Stable-Baselines3库实现
- **ALNS集成器**: 与外部ALNS算法进行交互
- **数据处理模块**: 读取和处理拥堵事件数据

### 2.2 依赖库

```python
import pandas as pd
import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C, DDPG, HER, SAC, TD3
import Intermodal_ALNS34959
import Dynamic_ALNS_RL34959
```

## 3. 强化学习实现分析

### 3.1 状态空间 (State Space)

#### 3.1.1 状态表示
状态空间使用 `Box` 空间定义，根据不同配置有两种形式：

- **基础版本** (2维): `Box(low=np.array([0, 0]), high=np.array([200, 6]))`
- **事件类型版本** (3维): `Box(low=np.array([0, 0, 0]), high=np.array([200, 6, 6]))`

#### 3.1.2 状态特征
状态向量在 `get_state()` 函数中构建 (lines 156-262)：

1. **延误容忍度** (`delay tolerance`): 0-200范围
2. **严重等级** (`severity level`): 1-6等级，基于拥堵持续时间计算
3. **事件类型** (`event type`): 0-6范围 (可选，当add_event_types=1时)

#### 3.1.3 严重等级计算
```python
if duration_length <= 20:
    severity_level = 1
elif duration_length <= 40:
    severity_level = 2
elif duration_length <= 60:
    severity_level = 3
elif duration_length <= 80:
    severity_level = 4
elif duration_length <= 100:
    severity_level = 5
else:
    severity_level = 6
```

### 3.2 动作空间 (Action Space)

#### 3.2.1 动作定义
动作空间为离散空间：`Discrete(2)`，包含两个动作：

- **动作 0**: 等待 (wait)
- **动作 1**: 执行 (go)

#### 3.2.2 动作执行机制
动作通过 `send_action()` 函数 (lines 103-154) 发送给ALNS系统：

1. 查找对应的`state_reward_pairs`条目
2. 将动作写入共享的数据结构
3. 等待ALNS系统处理并返回奖励

### 3.3 奖励函数 (Reward Function)

#### 3.3.1 奖励机制
奖励系统通过与ALNS的集成实现，主要有以下特点：

1. **外部奖励计算**: 奖励值由ALNS算法计算并返回
2. **延迟奖励**: 奖励在不确定性事件完成后给出
3. **奖励记录**: 所有奖励值保存在`all_rewards_list`中

#### 3.3.2 奖励获取流程
在 `step()` 函数中 (lines 284-504)：

```python
# 获取来自ALNS的奖励
for pair_index in Intermodal_ALNS34959.state_reward_pairs.index:
    if check:  # 检查匹配的不确定性索引、车辆、请求
        reward = Intermodal_ALNS34959.state_reward_pairs['reward'][pair_index]
        if type(reward).__module__ == 'numpy':
            reward = reward[0,0]
        all_rewards_list.append(reward)
```

## 4. 数据处理机制

### 4.1 数据读取

#### 4.1.1 拥堵数据文件路径
```python
data_path = "A:/MYpython/34959_RL/Uncertainties Dynamic planning under unexpected events/plot_distribution_targetInstances_disruption_" + duration_type + "_not_time_dependent" + "/R" + str(request_number_in_R) + "/Intermodal_EGS_data_dynamic_congestion" + str(table_number) + ".xlsx"
```

#### 4.1.2 数据结构
从Excel文件读取动态拥堵信息：
- `R_change_dynamic_travel_time`: 包含拥堵事件的详细信息
- 关键字段：`uncertainty_index`, `type`, `duration`, `event_types`

### 4.2 数据预处理

#### 4.2.1 持续时间处理
```python
duration = eval(R_change_dynamic_travel_time['duration'][index])
duration_length = duration[1] - duration[0]
```

#### 4.2.2 严重等级映射
将持续时间映射为6个严重等级，用于状态表示。

## 5. 训练流程

### 5.1 训练配置

#### 5.1.1 超参数设置
```python
episode_length = 1  # 回合长度
iteration_numbers_unit = 1  # 迭代单元
iteration_multiply = 1  # 迭代倍数
total_timesteps2 = iteration_numbers_unit * iteration_multiply
learning_starts = 10  # 学习开始步数
```

#### 5.1.2 算法选择
支持多种RL算法，主要使用DQN：
```python
for algorithm in ['DQN']:
    main(algorithm, mode)
```

### 5.2 训练循环

#### 5.2.1 学习-评估循环
```python
for number_of_learn_evaluate_loops in range(1000000000):
    model.learn(total_timesteps=total_timesteps2)  # 学习阶段
    evaluate = 1  # 切换到评估阶段
    # 评估模型性能
    average_reward, deviation = evaluate_policy(model, env, n_eval_episodes=iteration_numbers_unit)
```

#### 5.2.2 收敛判断
```python
if average_reward >= 0.9:
    sucess_times += 1
    if sucess_times >= 5:
        implement = 1  # 切换到实施阶段
```

### 5.3 经验回放

#### 5.3.1 状态-动作-奖励收集
```python
state_action_reward_collect = np.array(np.empty(shape=(0, 9)))
# 收集历史经验用于评估
state_action_reward_collect = np.vstack([state_action_reward_collect, add_row])
```

#### 5.3.2 经验重用
在评估阶段重用历史经验：
```python
for collect_index in list_of_collect_index:
    chosen_pair = state_action_reward_collect[collect_index]
    state = get_state(chosen_pair, ...)
    state_action_reward_collect_for_evaluate[state_key][chosen_pair[7]] = chosen_pair[8]
```

## 6. 实施阶段

### 6.1 实施触发条件
当训练满足以下条件时切换到实施阶段：
- 平均奖励 ≥ 0.9 连续5次
- 训练时间达到上限 (time_s ≥ 1)

### 6.2 实施流程
```python
while True:
    obs = env.reset()  # 重置环境
    action, _states = model.predict(obs)  # 预测动作
    implementation_time = timeit.default_timer() - implementation_start_time
    send_action(action[0])  # 发送动作到ALNS
```

### 6.3 实时性能监控
记录实施时间：
```python
append_new_line(implementation_time_path, str(implementation_time))
```

## 7. 关键功能函数

### 7.1 `get_state()` (lines 156-262)
**功能**: 根据输入数据生成状态表示
- 输入: `chosen_pair` 包含延误容忍度、经过终端、当前时间等信息
- 输出: 标准化的状态向量
- 特点: 处理事件类型和严重等级的映射

### 7.2 `send_action()` (lines 103-154)
**功能**: 将RL决策的动作发送给ALNS系统
- 机制: 通过共享的`state_reward_pairs`数据结构进行通信
- 同步: 确保动作被正确写入和读取

### 7.3 `stop_wait()` (lines 93-101)
**功能**: 处理系统中断和退出逻辑
- 检查退出标志文件 `34959.txt`
- 保存当前训练状态

### 7.4 `save_plot_reward_list()` (lines 60-91)
**功能**: 保存和可视化训练过程
- 计算滑动平均奖励
- 生成训练曲线图
- 保存奖励数据到文件

## 8. 系统集成

### 8.1 与ALNS的集成
- **数据共享**: 通过`Intermodal_ALNS34959.state_reward_pairs`进行数据交换
- **控制流程**: ALNS控制事件触发，RL提供决策
- **反馈机制**: ALNS计算奖励并反馈给RL

### 8.2 多进程协调
- 使用全局变量进行进程间通信
- 实现训练和实施的平滑切换
- 处理并发访问的同步问题

## 9. 性能优化

### 9.1 计算优化
- 使用CPU设备进行训练 (`device='cpu'`)
- 小批次学习 (`n_steps = 10`)
- 早期学习启动 (`learning_starts = 10`)

### 9.2 内存优化
- 及时清理已处理的数据记录
- 使用高效的数据结构存储经验

## 10. 监控和调试

### 10.1 性能监控
- 实时奖励记录
- 训练时间统计
- 实施时间测量

### 10.2 调试功能
- 详细的打印输出
- 状态-动作-对的完整记录
- 异常处理机制

## 11. 配置和扩展性

### 11.1 可配置参数
- `add_event_types`: 是否包含事件类型信息
- `wrong_severity_level_with_probability`: 错误严重等级的概率
- `time_dependent`: 是否使用时间相关特征

### 11.2 算法扩展性
- 支持多种RL算法 (DQN, PPO, A2C等)
- 可扩展的动作空间
- 灵活的状态表示

## 12. 应用场景

该系统主要应用于：
- **多式联运网络优化**: 处理卡车、火车、驳船等多种运输方式
- **动态拥堵管理**: 实时响应交通拥堵事件
- **不确定性决策**: 在信息不完全的情况下做出最优决策

## 13. 创新点

1. **RL-ALNS混合架构**: 结合强化学习和传统优化算法的优势
2. **动态实时决策**: 支持在线实时决策
3. **多状态表示**: 灵活的状态特征工程
4. **自适应学习**: 根据性能自动切换训练和实施阶段

## 14. 总结

`dynamic_RL34959.py` 实现了一个完整的强化学习系统来解决多式联运中的动态不确定性问题。该系统具有以下特点：

- **完整性**: 涵盖数据处理、模型训练、实时实施的完整流程
- **实用性**: 针对实际运输场景设计，具有很强的应用价值
- **扩展性**: 良好的模块化设计，便于扩展和修改
- **鲁棒性**: 完善的错误处理和异常恢复机制

该系统为动态运输网络优化提供了一个有效的解决方案，展示了强化学习在复杂物流系统中的应用潜力。