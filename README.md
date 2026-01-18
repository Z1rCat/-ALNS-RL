# 多式联运 ALNS+RL 项目（34959_RL）

## 项目概览
本项目基于 Adaptive Large Neighborhood Search（ALNS）结合强化学习（RL），用于动态多式联运场景中的不确定性事件处理与调度优化。ALNS 负责路径搜索与仿真，RL 负责在关键决策节点输出二元动作。

## 目录结构
```
.
├── codes/                          # 核心代码（ALNS、RL、调度器、脚本）
│   ├── Dynamic_master34959.py
│   ├── run_benchmark_replay.py
│   └── plot_paper_figure.py
├── codes/logs/run_*/               # 每次运行的日志与结果
│   ├── data/                       # 本次运行生成的不确定性事件数据（隔离）
│   ├── rl_trace.csv
│   ├── rl_training.csv
│   ├── rl_summary.csv
│   ├── baseline_wait.csv
│   ├── baseline_reroute.csv
│   └── paper_figures/              # 论文图表输出
├── distribution_config.json        # 分布配置（菜单自动读取）
├── ALNS_Research_Documentation/    # 文档与可视化脚本
│   ├── reports/
│   ├── analysis/
│   ├── config/
│   ├── scripts/
│   ├── data/
│   ├── figures/
│   └── figures_rl_logs/
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

## 推荐流程（端到端）
1. 运行实验：使用 `Dynamic_master34959.py` 生成 `run_*` 目录与日志。
2. 基准回放：用 `run_benchmark_replay.py` 生成 `baseline_*.csv`。
3. 论文图表：用 `plot_paper_figure.py` 输出到 `paper_figures/`。
4. 汇总分析：用 `ALNS_Research_Documentation/scripts/*.py` 汇总跨运行结果。

## 分布配置（distribution_config.json）
- 配置文件控制可选分布，主控面板自动读取，增删分布无需改代码。
- `means` 支持数值或对象写法：
  - `"A": 9`（默认正态分布）
  - `"A": {"mean": 9, "var": 4}`
  - `"A": {"mean": 9, "std": 2}`
  - `"A": {"mean": 9, "dist": "lognormal", "std": 2}`
- 详细说明见：`ALNS_Research_Documentation/config/Distribution_Config_Guide.md`。

## 输出与分析
- 运行日志：`codes/logs/run_*/rl_trace.csv`（动作级细节）。
- 训练统计：`codes/logs/run_*/rl_training.csv`（包含 `rolling_avg` 与 `recent_count`）。
- 结果汇总：`codes/logs/run_*/rl_summary.csv`。
- 运行数据：`codes/logs/run_*/data/`（本次生成的 Excel 不确定性事件）。
- 控制台输出：`codes/logs/run_*/console_output.txt`（完整运行日志）。
- 基准回放：`codes/logs/run_*/baseline_*.csv`（Always Wait/Reroute）。
- 论文图表：`codes/logs/run_*/paper_figures/`（PDF 输出）。

### 汇总与绘图
```bash
python ALNS_Research_Documentation/scripts/collect_rl_logs.py
python ALNS_Research_Documentation/scripts/plot_rl_logs_summary.py
```

### 论文级图表（单次运行）
```bash
python codes/plot_paper_figure.py --run-dir codes/logs/run_20260117_184322_R5_S0_Debug
```

### 基准策略回放
```bash
python codes/run_benchmark_replay.py --run-dir codes/logs/run_20260117_184322_R5_S0_Debug --policy wait
python codes/run_benchmark_replay.py --run-dir codes/logs/run_20260117_184322_R5_S0_Debug --policy reroute
```

## 部署提示
- 项目根目录自动推导，无需硬编码绝对路径。
- 生成数据与日志跟随 `run_*` 目录，支持并行运行的物理隔离。

## 收敛与跳级机制（简述）
- 收敛判定基于近期奖励滑动平均（`rolling_avg`），默认窗口长度为 30。
- 默认阈值为 0.7；调试场景 `S0_Debug` 使用 0.55 以便验证跳级逻辑。
- 跳级触发后会直接将 `table_number` 切换到下一阶段起点或测试起点（499）。

