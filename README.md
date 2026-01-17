# 多式联运 ALNS+RL 项目（34959_RL）

## 项目概览
本项目基于 Adaptive Large Neighborhood Search（ALNS）结合强化学习（RL），用于动态多式联运场景中的不确定性事件处理与调度优化。ALNS 负责路径搜索与仿真，RL 负责在关键决策节点输出二元动作。

## 目录结构
```
.
├── codes/                          # 核心代码（ALNS、RL、调度器、脚本）
├── codes/logs/run_*/               # 每次运行的日志与结果
│   ├── data/                       # 本次运行生成的不确定性事件数据（隔离）
│   ├── rl_trace.csv
│   ├── rl_training.csv
│   └── rl_summary.csv
├── distribution_config.json        # 分布配置（菜单自动读取）
├── ALNS_Research_Documentation/    # 文档与可视化脚本
└── Uncertainties Dynamic planning under unexpected events/  # 静态数据（只读）
```

## 快速开始
### 交互式运行（推荐）
```bash
python codes/Dynamic_master34959.py
```
按提示选择分布、R 值、运行轮数、算法与生成器核数。

### 命令行运行
```bash
python codes/Dynamic_master34959.py --dist_name S5_1 --request_number 30 --algorithm PPO --workers 4
python codes/Dynamic_master34959.py --dist_name S1_1 --request_number 10 --run_count 3 --single_core
```
参数说明：
- `--dist_name`: 分布名称，来自 `distribution_config.json`。
- `--request_number`: R 值（请求数量）。
- `--run_count`: 批量运行次数（>1 时会子进程执行）。
- `--algorithm`: `DQN` / `PPO` / `A2C`。
- `--workers`: 生成器进程数（`1` 为单核）。
- `--single_core`: 强制生成器使用单核。

## 分布配置（distribution_config.json）
- 配置文件控制可选分布，主控面板自动读取，增删分布无需改代码。
- `means` 支持数值或对象写法：
  - `"A": 9`（默认正态分布）
  - `"A": {"mean": 9, "var": 4}`
  - `"A": {"mean": 9, "std": 2}`
  - `"A": {"mean": 9, "dist": "lognormal", "std": 2}`
- 详细说明见：`ALNS_Research_Documentation/Distribution_Config_Guide.md`。

## 输出与分析
- 运行日志：`codes/logs/run_*/rl_trace.csv`（动作级细节）。
- 训练统计：`codes/logs/run_*/rl_training.csv`。
- 结果汇总：`codes/logs/run_*/rl_summary.csv`。
- 运行数据：`codes/logs/run_*/data/`（本次生成的 Excel 不确定性事件）。

### 汇总与绘图
```bash
python ALNS_Research_Documentation/collect_rl_logs.py
python ALNS_Research_Documentation/plot_rl_logs_summary.py
```

## 部署提示
- 项目根目录自动推导，无需硬编码绝对路径。
- 生成数据与日志跟随 `run_*` 目录，支持并行运行的物理隔离。

