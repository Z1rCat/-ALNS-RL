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
├── codes/logs/run_*/               # master 直跑输出目录
│   ├── data/                       # 本次运行生成的不确定性事件数据（隔离）
│   ├── rl_trace.csv
│   ├── rl_training.csv
│   ├── rl_summary.csv
│   ├── baseline_wait.csv
│   ├── baseline_reroute.csv
│   └── paper_figures/              # 论文图表输出
├── codes/nexus/<run_folder>/run_*/ # 服务器批量脚本输出目录（支持断点续跑）
├── distribution_config.json        # 分布配置（菜单自动读取）
├── ALNS_Research_Documentation/    # 文档与可视化脚本
│   ├── reports/
│   ├── latex/                      # 论文 LaTeX（可选，建议忽略编译中间产物）
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
- `--seed`: 随机种子（同时作用于数据生成与 RL）。
- `--run-name`: 覆盖默认的 run 目录名（用于脚本编排与并发）。

## 推荐流程（端到端）
1. 运行实验：使用 `Dynamic_master34959.py` 生成 `run_*` 目录与日志。
2. 基准回放：用 `run_benchmark_replay.py` 生成 `baseline_*.csv`。
3. 论文图表：用 `plot_paper_figure.py` 输出到 `paper_figures/`。
4. 汇总分析：用 `ALNS_Research_Documentation/scripts/*.py` 汇总跨运行结果。
5. 技术接口说明：见 `ALNS_Research_Documentation/reports/实现报告.md`（环境/日志/跳级/可扩展点）。

## 批量运行（本地/服务器）
### 服务器版本（可断点续跑）
```bash
python codes/experiments/run_experiments_server.py --run-folder server_batch_YYYYMMDD --max-workers 7
python codes/experiments/run_experiments_server_ppo.py --run-folder ppo_batch_YYYYMMDD --max-workers 7
```
可选参数：
- `--max-workers`: 覆盖默认并发数（默认=物理核数-2）。
- `--dry-run`: 只打印命令，不实际执行。
- `--run-folder`: 指定本批次输出目录（位于 `codes/nexus/` 下，或绝对路径）。
- `--no-precheck`: 关闭预检查（默认会执行断点预检查）。
- `--no-resume-existing`: 不续跑，强制新建 run。
- `--no-skip-completed`: 已完成任务也不跳过。

### 推荐：Transport 主实验调度脚本（Adaptive，无 baseline）
```bash
python codes/experiments/run_server_transport_main_suite.py \
  --run-folder transport_main_adaptive_notify \
  --wave smoke \
  --seed 42 \
  --request-number 30 \
  --max-workers 4 \
  --generator-workers 1 \
  --run-metrics \
  --run-plots \
  --cleanup-after-run
```
说明：
- 默认后端已切换为 `adaptive`，会根据历史运行时间在线调整任务调度顺序。
- 默认不跑 `baseline`，避免旧版回放阶段长期卡住。
- 成功 run 可自动清理 `data/` 与 `alns_outputs/`，但保留 `metrics.json`、`run_summary.json`、`paper_figures/` 等结果文件。
- `optimal_hybrid` 现在支持 `auto / gurobi / mp_bnb / serial_bnb` 四种求解路径：默认 `auto`，优先尝试 Gurobi，其次多进程分支定界，最后回退到串行分支定界。
- `transport_scheduler_warmstart_smoke_v1.json` 会作为默认 warm-start 模板；adaptive 运行中每完成一个任务，都会更新当前 `run_root` 下的状态文件，并可同步刷新模板供后续批次复用。

## 邮件通知系统（服务器运行提醒）
项目已经内置调度层通知系统，优先推荐用邮箱；短信通道复用 Twilio，QQ 推荐直接使用 QQ 邮箱 SMTP。

### 触发时机
- 启动通知：任务池建立后发送一次。
- 定时状态：默认每天 `08:00`、`12:00`、`16:00`、`20:00` 发送一次状态简报。
- 批次进展：默认每完成 `5` 个任务发送一次。
- 异常/重排：任务触发 timeout/stall/lock/unknown-retry 并被重排时发送。
- 完成通知：整批任务结束后发送。
- 单 run 失败：原有失败通知仍保留。

### 通知内容
邮件会同时发送：
- 中文纯文本简报
- 中文 HTML 表格版简报

内容包括：
- 运行目录、当前阶段、启动时间、当前时间、累计运行时间
- 总任务数、完成数、运行中、剩余、deferred、失败数、成功率
- CPU、内存、swap、可用内存、load/core
- ETA、预计完成时间、平均完成耗时、当前最长运行任务
- 按算法统计、按分布统计
- 当前运行中的任务
- 最近完成任务
- 最近重排/异常任务
- 最终失败任务
- 关键文件路径（events / summary / coef_state / live_status）
- 诊断提示（当前资源压力、是否存在长尾、当前调度求解模式等）

### 推荐依赖
邮件发送使用 Python 标准库 `smtplib`，不需要额外邮件库。  
建议安装 `psutil`，否则 CPU/内存/Swap 等资源指标会不完整：

```bash
python -m pip install psutil
```

若希望启用 Gurobi 作为调度器精确求解后端，还需要额外安装 `gurobipy`，并确保本机具备有效的 Gurobi License：

```bash
python -m pip install gurobipy
```

注意：`gurobipy` 不是默认必需依赖；未安装时，调度器会自动回退到 `mp_bnb` 或 `serial_bnb`。

### SMTP 环境变量
运行前设置以下环境变量即可启用邮箱通知：

- `EXP_NOTIFY_SMTP_HOST`
- `EXP_NOTIFY_SMTP_PORT`
- `EXP_NOTIFY_SMTP_USER`
- `EXP_NOTIFY_SMTP_PASSWORD`
- `EXP_NOTIFY_SMTP_FROM`
- `EXP_NOTIFY_SMTP_TO`
- `EXP_NOTIFY_SMTP_SSL`
- `EXP_NOTIFY_SMTP_TLS`
- `EXP_NOTIFY_COOLDOWN_S`

#### QQ 邮箱示例（PowerShell）
```powershell
$env:EXP_NOTIFY_SMTP_HOST="smtp.qq.com"
$env:EXP_NOTIFY_SMTP_PORT="465"
$env:EXP_NOTIFY_SMTP_USER="your_mail@qq.com"
$env:EXP_NOTIFY_SMTP_PASSWORD="你的SMTP授权码"
$env:EXP_NOTIFY_SMTP_FROM="your_mail@qq.com"
$env:EXP_NOTIFY_SMTP_TO="recv1@qq.com,recv2@outlook.com"
$env:EXP_NOTIFY_SMTP_SSL="1"
$env:EXP_NOTIFY_SMTP_TLS="0"
$env:EXP_NOTIFY_COOLDOWN_S="300"
```

注意：
- QQ 邮箱这里使用的是 SMTP 授权码，不是网页登录密码。
- `EXP_NOTIFY_SMTP_TO` 可以写多个收件人，用英文逗号分隔。
- 若测试时不希望“发送失败也进入 cooldown”，可临时设置：

```powershell
$env:EXP_NOTIFY_COOLDOWN_ON_SEND_FAIL="0"
```

### Twilio 短信通道（可选）
如需短信，设置：
- `EXP_NOTIFY_TWILIO_ACCOUNT_SID`
- `EXP_NOTIFY_TWILIO_AUTH_TOKEN`
- `EXP_NOTIFY_TWILIO_FROM`
- `EXP_NOTIFY_TWILIO_TO`

### 通知相关命令参数
`run_server_transport_main_suite.py` 会透传以下参数到 adaptive 调度器：

- `--notify-scheduler / --no-notify-scheduler`
- `--notify-schedule-times 08:00,12:00,16:00,20:00`
- `--notify-batch-size 5`
- `--notify-on-start / --no-notify-on-start`
- `--notify-on-requeue / --no-notify-on-requeue`
- `--notify-on-finish / --no-notify-on-finish`
- `--notify-live-status-interval-s 30`
- `--notify-success`
- `--no-notify-failure`

### 启用邮件通知的推荐命令
```powershell
& A:\MYpython\34959_RL\codes\env\python.exe A:\MYpython\34959_RL\codes\experiments\run_server_transport_main_suite.py `
  --run-folder transport_main_adaptive_notify `
  --wave smoke `
  --seed 42 `
  --request-number 30 `
  --max-workers 4 `
  --generator-workers 1 `
  --run-metrics `
  --run-plots `
  --cleanup-after-run `
  --notify-scheduler `
  --notify-schedule-times 08:00,12:00,16:00,20:00 `
  --notify-batch-size 5 `
  --notify-on-start `
  --notify-on-requeue `
  --notify-on-finish
```

### 仅测试邮件与调度器通知
如果只是想测试 QQ 邮箱和 adaptive 通知，不想真正训练，请直接运行 adaptive 后端自己的 `--dry-run`。  
注意：`run_server_transport_main_suite.py --dry-run` 只会打印后端命令，不会真正进入 adaptive 调度器，因此不会触发邮件发送。

```powershell
& A:\MYpython\34959_RL\codes\env\python.exe A:\MYpython\34959_RL\codes\experiments\run_experiments_server_adaptive.py `
  --run-folder transport_mail_test_direct `
  --variant PPO `
  --variant A2C `
  --dist-name M_60 `
  --dist-name R_10_120 `
  --request-number 30 `
  --seed 42 `
  --max-workers 2 `
  --generator-workers 1 `
  --scheduler-policy optimal_hybrid `
  --scheduler-opt-solver mp_bnb `
  --scheduler-opt-max-solver-workers 2 `
  --no-run-baseline `
  --no-run-plots `
  --no-run-metrics `
  --notify-scheduler `
  --notify-batch-size 1 `
  --notify-on-start `
  --notify-on-requeue `
  --notify-on-finish `
  --dry-run
```

### 最小真实调度测试命令
下面这条命令会真实运行 `PPO` 和 `A2C`，只覆盖两个分布，适合验证：
- 调度器是否按 `optimal_hybrid + mp_bnb` 工作
- 中文邮件是否正常
- 状态文件和模板是否会在运行中更新

```powershell
& A:\MYpython\34959_RL\codes\env\python.exe A:\MYpython\34959_RL\codes\experiments\run_experiments_server_adaptive.py `
  --run-folder ppo_a2c_mail_sched_test `
  --variant PPO `
  --variant A2C `
  --dist-name M_60 `
  --dist-name R_10_120 `
  --request-number 30 `
  --seed 42 `
  --max-workers 2 `
  --generator-workers 1 `
  --scheduler-policy optimal_hybrid `
  --scheduler-opt-solver mp_bnb `
  --scheduler-opt-max-solver-workers 2 `
  --template-state-path .\codes\experiments\transport_scheduler_warmstart_smoke_v1.json `
  --no-run-baseline `
  --run-metrics `
  --no-run-plots `
  --cleanup-after-run `
  --notify-scheduler `
  --notify-schedule-times "" `
  --notify-batch-size 2 `
  --notify-on-start `
  --notify-on-requeue `
  --notify-on-finish
```

### 运行中的状态文件
调度层会持续写出以下文件，便于邮件之外的人工排查：
- `codes/nexus/<run_folder>/adaptive_scheduler_events.csv`
- `codes/nexus/<run_folder>/adaptive_scheduler_coef_state.json`
- `codes/nexus/<run_folder>/adaptive_scheduler_summary.json`
- `codes/nexus/<run_folder>/adaptive_scheduler_live_status.json`
- `codes/nexus/<run_folder>/adaptive_scheduler_notify_state.json`

### 排错建议
- 若没有收到邮件，先检查 SMTP 环境变量是否在同一个终端会话中设置。
- 若资源指标显示 `n/a`，通常是未安装 `psutil`。
- 若邮件过于频繁，可调大 `EXP_NOTIFY_COOLDOWN_S` 或增大 `--notify-batch-size`。
- 若只想保留定时状态和结束通知，可关闭 `--notify-on-requeue`。
- 若运行在晚上首次启动，默认 `08:00,12:00,16:00,20:00` 的定时播报可能会在启动后集中补发；测试 QQ 邮箱时，建议把 `--notify-schedule-times ""` 置空，先只测启动/批次/完成邮件。
- 若出现 `email failed: Connection unexpectedly closed`，实验会继续运行，通知失败不会中断调度；当前版本不会自动补发失败的那一封邮件。

### 本地版本（调试用）
```bash
python codes/run_experiments_local.py
```
默认配置：`S5_1/S3_1/V1_3`、`R=30`、算法 `DQN+A2C`、`seed=42`。

## 分布配置（distribution_config.json）
- 配置文件控制可选分布，主控面板自动读取，增删分布无需改代码。
- `means` 支持数值或对象写法：
  - `"A": 9`（默认正态分布）
  - `"A": {"mean": 9, "var": 4}`
  - `"A": {"mean": 9, "std": 2}`
  - `"A": {"mean": 9, "dist": "lognormal", "std": 2}`
- 顶层 `variance` 支持标量或分阶段字典（用于 V1 变方差场景）。
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

### 批量重绘图（多次运行）
```bash
python codes/redraw_paper_figures.py --dry-run
python codes/redraw_paper_figures.py --clean --window 30
python codes/redraw_paper_figures.py --runs run_20260120_123246_371223_R30_V1_3_DQN_S42
```

### 基准策略回放
```bash
python codes/run_benchmark_replay.py --run-dir codes/logs/run_20260117_184322_R5_S0_Debug --policy wait
python codes/run_benchmark_replay.py --run-dir codes/logs/run_20260117_184322_R5_S0_Debug --policy reroute
python codes/run_benchmark_replay.py --run-dir codes/logs/run_20260117_184322_R5_S0_Debug --policy all
```

## 部署提示
- 项目根目录自动推导，无需硬编码绝对路径。
- 生成数据与日志跟随 `run_*` 目录，支持并行运行的物理隔离。

## 系统升级前备份（Git/GitHub）
建议在升级前完成一次“代码快照 + 运行结果离线备份”：

### 1) 提交并推送代码
```bash
git status
git add -A
git commit -m "chore: pre-upgrade checkpoint"
git push origin main
```

### 2) 给升级前状态打标签（可选但推荐）
```bash
git tag -a pre-upgrade-YYYYMMDD -m "pre system upgrade snapshot"
git push origin pre-upgrade-YYYYMMDD
```

### 3) 备份运行结果目录（重要）
`codes/logs/` 和 `codes/nexus/` 为大体积运行产物，默认不会推送到 GitHub。  
升级前请额外复制到外部磁盘或云盘（可先压缩）。

## 收敛与跳级机制（简述）
- 收敛判定基于近期奖励滑动平均（`rolling_avg`），默认窗口长度为 30，且需连续满足阈值若干次才算收敛。
- 默认阈值为 0.7；调试场景 `S0_Debug` 使用 0.3 以便快速验证跳级/切换逻辑。
- 训练阶段默认使用 `table_number=0..349`；测试阶段为 `table_number=499..350`（倒序，用于与训练阶段物理隔离）。
- 跳级触发由 ALNS 线程执行：会将 `table_number` 切换到下一阶段起点或测试起点（499），并切换 `implement=1`。

