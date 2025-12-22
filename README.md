# 多式联运 ALNS+RL 项目（34959_RL）

## 项目概览
本项目基于 Adaptive Large Neighborhood Search（ALNS）结合强化学习（RL），用于处理动态多式联运场景中的突发拥堵与不确定事件。`ALNS` 负责路径搜索与仿真，`RL` 负责在“规划/插入”两个关键阶段做出二元决策，从而提升实时调度的可行性与稳定性。

## 目录结构与说明

```
.
├── codes/                         # 核心代码（ALNS、RL、协调器、脚本）
├── codes/logs/run_*               # 实验运行生成的日志/图片（可再现）
├── ALNS_Research_Documentation/   # 文档与可视化资源
│   ├── rl_alns_run_report.md      # 本次组会汇报的 Markdown 报告
│   ├── rl_logs_aggregate.csv      # 跨运行的聚合统计表
│   ├── collect_rl_logs.py         # 采集日志并生成汇总的脚本
│   ├── plot_rl_logs_summary.py    # 读取聚合表并绘图的脚本
│   └── figures_rl_logs/           # 跨运行对比图（热力图 + 散点图）
└── Uncertainties Dynamic planning under unexpected events/
                                   # 由数据生成脚本产出的分布样本（忽略/不提交）
```

## 运行与复现建议

### 1. 数据生成
```powershell
& A:/MYpython/34959_RL/codes/env/python.exe a:/MYpython/34959_RL/codes/Dynamic_master34959.py
```
该脚本会生成数据、调用 ALNS+RL 双线程，并输出 `codes/logs/run_*` 文件夹，记录详尽的 `rl_training.csv`、`rl_trace.csv` 与图片。

### 2. 结果整理与绘图
```powershell
& A:/MYpython/34959_RL/codes/env/python.exe a:/MYpython/34959_RL/ALNS_Research_Documentation/collect_rl_logs.py
& A:/MYpython/34959_RL/codes/env/python.exe a:/MYpython/34959_RL/ALNS_Research_Documentation/plot_rl_logs_summary.py
```
第一步汇总所有 `run_*` 日志到 `rl_logs_aggregate.csv`；第二步读取聚合表绘制：
- `figures_rl_logs/overall_implement_avg_reward_heatmap.png`
- `figures_rl_logs/overall_reward_vs_action_ratio.png`

### 3. 快速查阅报告
`ALNS_Research_Documentation/rl_alns_run_report.md` 包含：
- 系统架构概览
- RL 调用流程与 reward 释义
- 不同分布/R 下的热图与动作分析
- 典型弱/强/中性能例证与图片

## 其他说明

- 强化学习训练数据在 `codes/logs/run_*/rl_training.csv`，其中 `phase=implement` 的 `reward` 即用于实施阶段直方图与正确率评估。
- `codes/logs/run_*/rl_trace.csv` 记录了每一次 `begin_removal` / `begin_insertion` 的动作、reward，以及阶段标记，便于追踪 RL 在哪个阶段被调用。
- `codes/logs/run_*/rl_summary.csv` 汇总了 `average_reward`、`removal`/`insertion` 动作数，常用于跨运行对比。

## 版本管理与忽略策略
本项目采用 `.gitignore` 忽略虚拟环境、日志与自动生成的数据（详见 `.gitignore`），确保仓库只跟踪源码与核心文档。
