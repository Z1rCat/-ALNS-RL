# Codes 目录主入口（精简版）

为避免重复脚本和参数分叉，`codes/` 现在只保留以下主入口思路：

1. `codes/experiments/run_experiments_server_unified.py`
- 统一训练入口。
- 支持 `PPO` / `PPO_NEW` 各版本（`v1/v2/v3/v3.1/v3.2/v4.1/v4.2_*/v4.3_*`）。
- 支持多分布、多 seed、多请求数、多核心并行（`--max-workers`）。

2. `codes/experiments/run_main_pipeline.py`
- 一键流水线入口（推荐）。
- 自动执行：训练 -> probe 汇总 -> phase 汇总 -> 汇总图绘制。
- 适合论文/报告所需的批量对比产物生成。

3. `codes/experiments/run_experiments_server_stream.py`
- 流式任务调度入口（新增）。
- 按 `(variant, dist, R, seed)` 任务级队列并行。
- 某个任务一结束立即补下一个任务，持续维持 `--max-workers` 并发，不再“按算法整批等待”。

4. `codes/experiments/run_benchmark_replay.py`
- 对既有 `run_*` 目录做 baseline replay（wait/reroute/random）补算。

## 最常用命令

### 1) 仅跑训练（统一入口）
```bash
python codes/experiments/run_experiments_server_unified.py ^
  --run-folder ppo_main_runs ^
  --variant PPO ^
  --variant PPO_NEW:v1 ^
  --variant PPO_NEW:v2 ^
  --variant PPO_NEW:v3 ^
  --dist-name O_10_90 ^
  --dist-name O_10_60 ^
  --dist-name G_10_30_60 ^
  --dist-name G_10_60_90 ^
  --request-number 30 ^
  --seed 42 ^
  --max-workers 6 ^
  --no-run-baseline ^
  --no-run-plots ^
  --run-metrics
```

### 2) 一键训练+汇总+绘图（推荐）
```bash
python codes/experiments/run_main_pipeline.py ^
  --run-folder ppo_main_runs ^
  --report-folder ppo_main_report ^
  --variant PPO ^
  --variant PPO_NEW:v1 ^
  --variant PPO_NEW:v2 ^
  --variant PPO_NEW:v3 ^
  --dist-name O_10_90 ^
  --dist-name O_10_60 ^
  --dist-name G_10_30_60 ^
  --dist-name G_10_60_90 ^
  --request-number 30 ^
  --seed 42 ^
  --max-workers 6
```

### 3) 流式任务并发（任务完成即补位）
```bash
python codes/experiments/run_experiments_server_stream.py ^
  --run-folder thesis_gapfill_patch ^
  --variant A2C ^
  --variant PPO ^
  --variant PPO_NEW:v2 ^
  --variant PPO_NEW:v3 ^
  --variant PPO_NEW:v3.1 ^
  --dist-name F1_10_90 ^
  --dist-name G_10_90_60 ^
  --dist-name O_10_60 ^
  --dist-name O_10_90 ^
  --request-number 30 ^
  --seed 42 ^
  --seed 2026 ^
  --seed 3333 ^
  --max-workers 8 ^
  --no-run-baseline ^
  --no-run-plots ^
  --run-metrics ^
  --skip-completed ^
  --no-resume-existing ^
  --no-precheck
```

## 一键流水线输出

默认输出到 `codes/nexus/<report-folder>/`：

- `probe_summary.csv`
- `<summary_prefix>_summary.csv`（默认 `phase_main_summary.csv`）
- `<summary_prefix>_summary.md`
- `<summary_prefix>_summary.html`
- `plot_avg_reward_by_dist_variant.png`
- `plot_action1_rate_vs_avg_reward.png`
- `plot_probe_delta_bacc_by_variant.png`
- `probe_reports/*.json`
- `pipeline_manifest.json`

运行日志与单次实验目录在 `codes/nexus/<run-folder>/run_*` 下。
