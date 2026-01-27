# 运行入口与目录结构说明

本文件用于说明 `codes/` 目录的主入口脚本、子目录功能以及常用运行方式。当前结构经过整理，**根目录只保留少量入口脚本**，其余功能模块均放入分类子目录。

## 一、根目录主入口（建议从这里启动）

以下脚本是团队常用入口，位于 `codes/` 根目录：

- `codes/Dynamic_master34959.py`  
  主运行入口（单次实验/训练/实施），支持算法选择（DQN/PPO/A2C/PPO_HAT/A2C_HAT/LBKLAC 等）。

- `codes/run_experiments_server.py`  
  服务器批量实验入口（当前配置：R=30、seed=42、算法含 DQN/PPO/A2C/PPO_HAT）。

- `codes/run_experiments_local.py`  
  本地批量实验入口（小规模调试用）。

- `codes/run_smoke_server.py`  
  冒烟测试入口（快速验证环境）。

## 二、子目录分类与用途

- `codes/core/`  
  核心算法与双线程系统核心逻辑（ALNS/RL 协调与环境实现）
  - `Dynamic_ALNS_RL34959.py`
  - `dynamic_RL34959.py`
  - `Intermodal_ALNS34959.py`
  - `rl_logging.py`
  - `config.py`
  - `emission_models.py`
  - `fuzzy_HP.py`

- `codes/experiments/`  
  批量实验调度与基线 replay
  - `run_experiments_common.py`
  - `run_experiments_server.py`
  - `run_experiments_local.py`
  - `run_smoke_server.py`
  - `run_benchmark_replay.py`

- `codes/analysis/`  
  指标与统计分析
  - `compute_metrics.py`（会生成 `metrics.json` 和汇总 `metrics_summary.csv`）
  - `calc_run_durations.py`

- `codes/plotting/`  
  绘图相关
  - `plot_paper_figure.py`
  - `redraw_paper_figures.py`

- `codes/generation/`  
  数据生成与事件模拟
  - `generate_mixed_parallel.py`
  - `generate_un_expected_events_by_stochastic_info_RL.py`

- `codes/robust_rl/`  
  鲁棒强化学习相关模型（LBKLAC、HAT 等）

- `codes/tools/`  
  工具类脚本（例如 run 清理）
  - `cleanup_run.py`

- `codes/logs/`  
  每次运行的日志与输出（`run_*` 子目录）

## 三、常用运行方式（示例）

### 1) 单次实验
```bash
python codes/Dynamic_master34959.py --dist_name S5_1 --request_number 30 --algorithm PPO_HAT --seed 42 --workers 1
```

### 2) 服务器批量实验
```bash
python codes/run_experiments_server.py --max-workers 32
```

### 3) 本地批量实验
```bash
python codes/run_experiments_local.py --max-workers 2
```

### 4) 基线 replay（对已有 run 目录）
```bash
python codes/experiments/run_benchmark_replay.py --run-dir codes/logs/run_xxx --policy all --include-random
```

### 5) 计算指标（G 指标等）
```bash
python codes/analysis/compute_metrics.py --run-dir codes/logs/run_xxx
```

### 6) 绘图
```bash
python codes/plotting/plot_paper_figure.py --run-dir codes/logs/run_xxx
python codes/plotting/redraw_paper_figures.py
```

## 四、关于路径与导入（重要）

- 运行入口脚本会自动把 `codes/` 加入 `sys.path`，所以核心模块在 `codes/core/` 中以 `from core import ...` 方式统一导入。
- 如需自定义脚本，请确保在脚本开头加入：
  ```python
  from pathlib import Path
  import sys
  CODES_DIR = Path(__file__).resolve().parent if "codes" in str(Path(__file__).resolve()) else Path("codes").resolve()
  if str(CODES_DIR) not in sys.path:
      sys.path.insert(0, str(CODES_DIR))
  ```

## 五、快速排查建议

- 运行失败时，优先查看 `codes/logs/run_*/console_output.txt`
- 基线卡死/超时会在 `run_*/watchdog_events.jsonl` 记录
- 失败 run 会生成 `run_*/FAILED.json`
