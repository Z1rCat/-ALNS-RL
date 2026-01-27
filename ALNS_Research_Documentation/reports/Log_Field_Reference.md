# 运行日志字段对照表

本文档说明运行目录 `codes/logs/run_YYYYMMDD_.../` 下主要日志文件的字段含义。

## 1) rl_trace.csv / baseline_*.csv（动作轨迹）
`baseline_wait.csv` / `baseline_reroute.csv` 的字段与 `rl_trace.csv` 完全一致。

| 字段 | 含义 |
| --- | --- |
| ts | 时间戳（Unix 秒） |
| phase | 阶段：`train` 或 `implement` |
| stage | 轨迹事件类型，如 `send_action` / `receive_reward` / `finish_removal` / `finish_insertion` |
| uncertainty_index | 不确定事件索引 |
| request | 请求/订单 ID |
| vehicle | 车辆 ID |
| table_number | 环境文件序号（0-499） |
| dynamic_t_begin | 动态事件开始时间（ALNS 内部变量） |
| duration_type | 不确定性类型（ALNS 内部变量） |
| gt_mean | 环境分布均值（来自生成器 `__meta__`） |
| phase_label | 环境阶段标签（A/B/C） |
| delay_tolerance | 允许延迟时间 |
| severity | 拥堵严重度（离散等级） |
| passed_terminals | 已经过的终端序列 |
| current_time | 当前仿真时间 |
| action | 动作：0=等待/保持，1=重规划；插入阶段 0=接受/1=拒绝；`-10000000` 表示占位 |
| reward | 奖励值；`-10000000` 表示占位 |
| action_meaning | 动作语义（如“等待/保持”“重新规划”“接受插入”“拒绝插入”） |
| feasible | 可行性标记（如有） |
| source | 来源：`RL` 或 `BASELINE` |

说明：
- `baseline_*.csv` 是基准策略回放输出，`source` 标记为 `BASELINE`。
- `stage` 与 `action_meaning` 主要用于区分 removal / insertion 两类决策。

## 2) rl_training.csv（训练过程）

| 字段 | 含义 |
| --- | --- |
| ts | 时间戳（Unix 秒） |
| phase | `train` / `eval` / `implement` |
| step_idx | 全局步数计数 |
| reward | 训练/实施阶段的即时奖励 |
| avg_reward | 评估均值（仅 `eval` 行有） |
| std_reward | 评估标准差（仅 `eval` 行有） |
| rolling_avg | 最近窗口（默认 30）奖励均值 |
| recent_count | 窗口内样本数 |
| training_time | 训练累计耗时（秒） |
| implementation_time | 实施阶段累计耗时（如有） |

说明：
- `eval` 行用于跳级判定：`rolling_avg` 连续达到阈值才触发。

## 3) rl_summary.csv（实施阶段统计）

| 字段 | 含义 |
| --- | --- |
| ts | 时间戳（Unix 秒） |
| reward_count | 统计样本数量 |
| average_reward | 平均奖励 |
| std_reward | 奖励标准差 |
| removal_action | removal 操作的动作次数 |
| removal_action_reward | removal 操作平均奖励 |
| removal_wait_action | removal 等待次数 |
| removal_wait_action_reward | removal 等待奖励 |
| insertion_action | insertion 接受次数 |
| insertion_action_reward | insertion 接受奖励 |
| insertion_non_action | insertion 拒绝次数 |
| insertion_non_action_reward | insertion 拒绝奖励 |

## 4) meta.json（运行配置）

| 字段 | 含义 |
| --- | --- |
| distribution | 分布名称（如 S5_1） |
| request_number | R 值（请求数量） |
| generator_workers | 生成器并发数 |
| algorithm | RL 算法（DQN/PPO/A2C） |
| seed | 随机种子 |
| run_name | 本次运行目录名 |
| stop_flag_file | 停止标志文件路径 |
| data_root | 数据输出根目录 |
