# ALNS + RL（SB3）动态多式联运：面向不确定性/分布漂移的鲁棒训练框架（可插拔/可评估/可落地）

本文基于本仓库**真实代码路径**梳理现状，并给出一个“最小侵入、接口可插拔、日志可评估”的新算法/训练框架落地方案。重点面向：**非平稳/分布漂移、ALNS 步极贵、动作空间小（通常 2）**的现实约束。

---

## 现状系统梳理（带文件名与关键函数/变量）

### 1) 总体调度与 run_dir 产物

- 主入口：`codes/Dynamic_master34959.py`
  - `run_single()`：创建 `codes/logs/run_YYYYMMDD_.../`（通过 `codes/rl_logging.py:set_run_dir()`），写入 `meta.json`（`rl_logging.write_meta()`），设置环境变量：
    - `SCENARIO_NAME`、`RL_ALGORITHM`、`RL_SEED`
    - `STOP_FLAG_FILE`（run_dir 下的 `34959.txt`）
    - `DYNAMIC_DATA_ROOT`（run_dir/data）
    - `ALNS_OUTPUT_ROOT`（run_dir）
  - `run_generator()`：调用 `codes/generate_mixed_parallel.py` 生成 500 个 Excel 环境文件到 `run_dir/data/`
  - `run_simulation()`：用 `ThreadPoolExecutor` 启动 **2 个线程**并行运行 `codes/Dynamic_ALNS_RL34959.py:main()`，形成 “ALNS 线程 + RL 线程”

### 2) RL 的真实形态：Gym Env + 线程共享变量（不是纯 Gym 也不是纯 RPC）

结论：RL 在系统里是 **Gym 环境（`coordinationEnv`）+ SB3 训练**，但 `Env.reset()/step()` 的语义由 **线程共享变量（DataFrame）驱动的阻塞式同步**决定。

- Gym 环境：`codes/dynamic_RL34959.py`
  - `class coordinationEnv(Env)`：`action_space = Discrete(2)`，`observation_space = Box(...)`
  - `reset()`：**轮询** `Intermodal_ALNS34959.state_reward_pairs`，找到 `action == -10000000` 的行作为当前决策点
  - `step(action)`：
    1) `send_action(action)` 写回共享表 `state_reward_pairs['action']`
    2) 继续轮询共享表直到 `reward != -10000000`，将其作为本步 reward
    3) 写入 `rl_training.csv`（`log_training_row()`）与 `rl_trace.csv`（`log_trace_from_row(..., stage="receive_reward")`）

> 因为 `episode_length = 1`（见 `dynamic_RL34959.main()`），所以在 ALNS 模式下本质上是“**单步 episode**”：每次不确定事件触发一个决策（上下文 bandit 化非常明显）。

### 3) ALNS 与 RL 的衔接点（同步机制与阶段切换）

- 线程划分：`codes/Dynamic_master34959.py:run_simulation()`
  - `approach=0`：ALNS 主仿真线程（调用 `Dynamic_ALNS_RL34959.Intermodal_ALNS_function()`）
  - `approach=1`：RL 训练/推理线程（调用 `dynamic_RL34959.main(RL_ALGORITHM, 'barge')`）

- 共享状态/硬耦合点（关键全局变量）：
  - 共享表：`codes/Intermodal_ALNS34959.py:real_main()` 初始化 `state_reward_pairs`（列包括 `uncertainty_index/uncertainty_type/request/vehicle/delay_tolerance/passed_terminals/current_time/action/reward`）
  - 信号标志：
    - `dynamic_RL34959.implement`（训练/测试阶段开关）
    - `dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase`（切换期间让 RL stop）
    - `Intermodal_ALNS34959.ALNS_implement_start_RL_can_move`（实现/测试阶段：ALNS 通知 RL “可以决策了”）
    - `dynamic_RL34959.clear_pairs_done`、`dynamic_RL34959.ALNS_got_action_in_implementation`（实现阶段 handshake）
  - 表号驱动：`Dynamic_ALNS_RL34959.table_number`（写入 trace，且决定读取哪个 Excel）

### 4) curriculum jumping（智能跳级）在代码里的真实位置

- 实现位置：`codes/Dynamic_ALNS_RL34959.py:main()` 的 while 循环内
  - 收敛判据由 RL 线程写入的全局变量提供：`dynamic_RL34959.curriculum_converged`
  - 跳级逻辑：根据 `distribution_config.json` 的 `pattern`（`load_distribution_patterns()`）决定跳转（例如 S5 adaptation：先跳到 100，再满足条件跳到 test=499）
  - 强制切换：达到 `table_number >= 349` 时强制进入实现/测试（设置 `dynamic_RL34959.implement = 1` 且 `next_table_number = 499`）

### 5) 数据生成与漂移“可得性”

- 分布配置：`distribution_config.json`
  - `pattern`：`ab/aba/abc/recall/adaptation/random_mix`
  - `variance` 与 `extra.p_disaster`（mixture disaster）等信息在这里定义
- 生成器：`codes/generate_mixed_parallel.py`
  - 对每个 `table_number` 写入 Excel 的 `__meta__` sheet：包含 `gt_mean` 与 `phase_label`
  - 当前 **没有**把 `gt_std` / `p_disaster` 写进 `__meta__`
- RL 侧读取漂移信息：`codes/dynamic_RL34959.py:get_state()`
  - 从 Excel `__meta__` sheet 读取并写入全局：`current_gt_mean`、`current_phase_label`
  - 这些字段随后被写入 `rl_trace.csv` 的 `gt_mean/phase_label` 列（见 `log_trace_from_row()`）

---

## RL 接入点与数据流图（文字描述）

**线程 A（ALNS）**：`Dynamic_master34959.py` → `Dynamic_ALNS_RL34959.main(approach=0)` → `Intermodal_ALNS34959.real_main()`  
当检测到不确定事件并需要 RL 决策时：
1) ALNS 往 `Intermodal_ALNS34959.state_reward_pairs` append 一行（`action=-10000000, reward=-10000000`）并写 trace：`Intermodal_ALNS34959.log_rl_event(..., stage="begin_removal"/"begin_insertion")`
2) ALNS 在实现阶段设置 `Intermodal_ALNS34959.ALNS_implement_start_RL_can_move = 1` 作为“允许 RL 决策”的信号

**线程 B（RL）**：`Dynamic_ALNS_RL34959.main(approach=1)` → `dynamic_RL34959.main()`  
1) `coordinationEnv.reset()` 轮询 `state_reward_pairs`，拿到一条待决策 state（并同步全局 `uncertainty_index/vehicle/request`）
2) 算法输出 action（SB3 的 `model.predict(obs)` 或本文新增的 `DRCB`）
3) `send_action(action)` 将 action 写回共享表，并写 trace：`stage="send_action"`
4) `coordinationEnv.step()` 轮询 reward 回填（由 ALNS 计算后写入），写 trace：`stage="receive_reward"`；写 training：`rl_training.csv`

**实现/测试阶段**：`dynamic_RL34959.implement=1` 后，RL 线程不再调用 `model.learn()`，而是等 `ALNS_implement_start_RL_can_move` 信号后直接 `predict` 并 `send_action`（见 `dynamic_RL34959.main()` 的实现段循环）。

---

## 环境与训练循环（工程级细节）

### (A) 环境类与接口

- 环境类：`codes/dynamic_RL34959.py:coordinationEnv`
- 构造：`env = coordinationEnv()`（`dynamic_RL34959.main()` 与 `run_benchmark_replay.py` 都这么做）

**reset()**
- 行为：阻塞式轮询 `Intermodal_ALNS34959.state_reward_pairs`，找到 `action == -10000000` 的行作为“当前要决策的不确定事件”
- 返回 `obs`：
  - 默认 `add_event_types == 0`：`obs.shape == (2,)`，dtype=float
  - 若 `add_event_types == 1`：`obs.shape == (3,)`
  - 真实构造：`get_state(chosen_pair)`，核心维度：
    - `obs[0] = delay_tolerance`
    - `obs[1] = severity_level`（由 duration_length 分段得到 1..6）
    - `obs[2] = event_type`（仅当 add_event_types=1）
  - 漂移信息：`gt_mean/phase_label` 会被读入全局（`current_gt_mean/current_phase_label`），但**不进入 obs**（仅进入日志）

**step(action)**
- 动作语义（你已给出，并与代码一致）：
  - `action=0`：wait/keep（等待/保持）
  - `action=1`：reroute（重规划/绕行）
- reward：RL 侧不计算；由 ALNS 写入 `state_reward_pairs['reward']` 后，Env 轮询到 `reward != -10000000` 才返回
- done：`episode_length == 1` 时每步 done=True（ALNS 模式下单步 episode）
- info：当前实现基本为空 `{}`（可作为“插漂移信号”的低侵入扩展点）

**implement/test 如何进入与 table_number 逻辑**
- 训练/测试边界来自生成器的分段定义（例如 `pattern="ab"`：0..349 train，350..499 implement），但系统允许跳级：
  - `Dynamic_ALNS_RL34959.py` 达到收敛阈值会将 `table_number` 跳到 499，并置 `dynamic_RL34959.implement = 1`
  - 测试阶段实际是从 499 往下跑到 350（见 `Dynamic_ALNS_RL34959.py`：implement 时 `table_number -= 1`，到 `<350` 结束）

### (B) SB3 的使用方式（当前实现）

- 创建位置：`codes/dynamic_RL34959.py:main()`
  - DQN：`DQN('MlpPolicy', env, learning_starts=10, device='cpu', seed=...)`
  - PPO/A2C：`n_steps=10, device='cpu', seed=...`
- `total_timesteps` 的含义：
  - `total_timesteps2 = iteration_numbers_unit * iteration_multiply`，默认都是 1
  - 因为 `episode_length=1`，所以在 ALNS 模式下：**1 个 timesteps ≈ 1 次不确定事件决策**（不等于 1 个 table_number；一个 table 可能触发多次决策）
- 并行：
  - 本项目并行主要是 “同一 run 内：ALNS 线程 + RL 线程”
  - 多 run 并发通过外部脚本（例如 `run_experiments_common.py`）实现

### (C) 跳级/漂移/日志钩子

- curriculum：
  - 判据计算：`dynamic_RL34959.py` 里用 `recent_rewards` 的 rolling_avg 与 `CURRICULUM_REWARD_THRESHOLD/CURRICULUM_SUCCESS_REQUIRED`
  - 跳级执行：`Dynamic_ALNS_RL34959.py` 根据 `distribution_config.json` 的 `pattern` 决定跳转点
- 漂移可视化：
  - `gt_mean/phase_label` 来自 Excel `__meta__`（生成器写入），RL 在 `get_state()` 读取并写入 trace
- `rl_trace.csv` 的 stage：
  - RL 写：`send_action`、`receive_reward`（`dynamic_RL34959.log_trace_from_row`）
  - ALNS 写：`begin_removal/finish_removal/begin_insertion/finish_insertion`（`Intermodal_ALNS34959.log_rl_event`）
- baseline 回放：
  - `codes/run_benchmark_replay.py` 用原 run 的 `rl_trace.csv` 推导 `(table_number, phase)` 序列，逐表回放 Always_Wait/Always_Reroute/Random，输出 `baseline_*.csv`

### (D) “可观测漂移指标”插入位置（建议）

现状信号来源：
- supervised：漂移切换点可由 `phase_label` 或 `table_number` 分段直接给出（生成器已知）
- unsupervised：可用在线统计（reward/obs 分布、残差）推断

建议工程落点（兼容 RL 与 baseline）：
1) **trace 统一字段（推荐）**：把 drift 信号写进 `rl_trace.csv` 的新增列（不会破坏 `compute_metrics.py`，只增量）
2) **info 透传（可选）**：在 `Env.step()` 的 `info` 放 `drift_score/regime_id` 供算法使用，但需要 SB3 callback 才能统一落日志
3) **独立 drift monitor 模块（推荐）**：算法与日志都通过同一个 DriftMonitor 读同一份 meta（Excel `__meta__` + `distribution_config.json`）

> 已做的最小落地：`codes/dynamic_RL34959.py` 与 `codes/Intermodal_ALNS34959.py` 的 trace/training 追加了 `algo/regime_id/context_id/drift_score` 字段（可为空，不影响旧评估链）。

---

## 新算法可插拔 API 设计（函数签名 + 映射说明）

### (A) 代码组织建议（模块化）

建议新增（或已新增）：
- `codes/robust_rl/`
  - `drcb.py`：**新算法 DRCB**（Drift-Robust Contextual Bandit），无需 torch/SB3，适合 ALNS 步极贵 + 单步决策
  - （后续可扩展）`sb3_adapter.py`、`drift_monitor.py`、`trainer.py`

建议“只改少量”的 existing 文件：
- `codes/Dynamic_master34959.py`：允许 `--algorithm DRCB`（已加）
- `codes/dynamic_RL34959.py`：在模型创建处增加 `algorithm == 'DRCB'` 分支（已加），并在日志里追加 drift 字段（已加）

### (B) 统一接口（最小抽象层）

推荐在新模块中逐步收敛为以下抽象（本项目现状可以先“鸭子类型”适配）：

```python
class Agent:
    def act(self, obs, *, deterministic: bool, context: dict) -> int: ...
    def observe(self, obs, action, reward, next_obs, done, info) -> None: ...
    def update(self) -> dict: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path, **kwargs) -> "Agent": ...
```

映射到现有系统：
- “act” 对应：`dynamic_RL34959.main()` 的 `model.predict(obs)`（实现阶段）或 `model.learn(...)` 内部 rollout（训练阶段）
- “observe/update” 对应：当前由 SB3 内部完成；对自研算法可在 `learn()` 内显式调用
- “日志”对齐：继续使用 `dynamic_RL34959.log_trace_from_row` 与 `log_training_row`（字段不变/仅增量）

---

## 推荐的新算法方向（最多 2 个）+ 为什么

### 方向 1（推荐，已落地雏形）：DRCB（Drift-Robust Contextual Bandit）

代码：`codes/robust_rl/drcb.py:DriftRobustContextualBandit`  
启用：`Dynamic_master34959.py` 选择算法 `DRCB`（或 `--algorithm DRCB`）

核心思想（适配本项目约束）：
- ALNS 模式下 `episode_length=1` ⇒ 近似 contextual bandit
- 用线性模型分别拟合每个 action 的期望 reward：`E[r|x,a]`
- 非平稳/漂移处理：
  - **指数遗忘（decay）**：对旧样本自动降权，快速适应新分布
  - **按 regime 分桶**（默认用 `phase_label`）：每个 phase 独立估计（A/B/C），抑制“混合训练导致的互相污染”
  - **UCB 探索**（ucb_alpha）：在样本极少时稳健探索，避免过拟合

性价比：
- 不依赖 torch，训练成本低，单步决策场景通常更稳定/更可解释
- 适合 “步很贵、不允许大量探索” 的在线学习

### 方向 2（建议后续评估）：Drift-aware Recurrent PPO + 近期样本加权 + 防遗忘

适用场景：
- 如果你确认 action 对后续 ALNS 多步有长期影响（而不是“几乎即时结算”），才值得上 RNN / meta-RL

工程建议：
- 仅做**小改造**：给 policy 增加 context（`gt_mean/phase_label` 或在线统计），并在 replay/rollout 上做 recency weighting
- 防遗忘：对旧 regime 的少量样本做 rehearsal（每轮保留少量 A/B/C 样本）

不建议（当前性价比低）：
- 复杂 meta-RL / 大规模 domain randomization：ALNS step 太贵，探索成本承受不了
- 需要大量并行环境的算法：本项目主要靠“多 run 并发”，单 run 内并行环境收益不高

---

## 落地改造清单（按文件列出要改的点）

已落地（本次提交）：
- `codes/dynamic_RL34959.py`
  - trace/training 增量字段：`algo/regime_id/context_id/drift_score`
  - 新算法分支：`algorithm == "DRCB"` → `robust_rl.DriftRobustContextualBandit`
- `codes/Intermodal_ALNS34959.py`
  - 与 RL 对齐的 trace 字段增量，并尝试复用 `dynamic_RL34959._drift_snapshot()` 保持一致
- `codes/Dynamic_master34959.py`
  - 允许选择 `DRCB`
- `codes/run_experiments_server.py`
  - 实验配置允许跑 `DRCB`
- 新增：`codes/robust_rl/drcb.py`（算法实现）

建议后续（不破坏评估链条的前提下）：
1) `codes/generate_mixed_parallel.py`：在 `__meta__` 增加 `gt_std/p_disaster`（对应 distribution_config 的 variance/extra）
2) `codes/run_benchmark_replay.py`：把 drift 字段也写到 baseline（现在字段存在但多数为空）
3) 用 timeout + backoff 替换 busy loop（`send_action/reset/step` 的 while True），避免卡死风险

---

## 评估与验收清单（如何判定比 PPO/DQN/A2C 好）

沿用现有基础设施（不改口径）：
- 指标实现：`codes/compute_metrics.py`
  - 仅统计：`phase == "implement"` 且 `stage == "receive_reward"`（RL 轨迹）
  - baseline 用同口径（优先 `receive_reward`，否则 `finish_removal`）
  - 若 `n_test` 不一致：直接报错（不允许补齐）
- 关键指标：
  - `Adv0=(R_RL-R_wait)/n`
  - `Adv1=(R_RL-R_reroute)/n`
  - `AdvRand=R_rand/n`
  - `G=(Adv0+Adv1)-AdvRand`

验收步骤（建议固定模板）：
1) 同一 `(dist, R, seed)` 下跑 `DQN/PPO/A2C/DRCB`，并跑 baseline（wait/reroute/random）
2) 对每个 run，确保：
   - `rl_trace.csv` 存在且 implement 段的 `receive_reward` 行数一致
   - `baseline_*.csv` 与 RL 的 table/phase 序列一致（`run_benchmark_replay.py` 已强约束）
3) 用 `compute_metrics.py` 出 `metrics.json`，比较 `G` 的均值/方差与显著性
4) 用 `plot_paper_figure.py` 生成 Fig2/Fig4：
   - 左轴：reward/advantage
   - 右轴：`gt_mean/gt_std/p_disaster/risk quantile`（按 `distribution_config.json` 选择）
   - 解释机制：结合新增列 `drift_score/context_id/regime_id`

---

## 风险清单（死锁、对齐失败、日志缺失、收敛误判等）与规避方案

1) **死锁/卡死（busy wait）**
   - 现状：`reset()/step()/send_action()` 大量 `while True` 轮询共享表
   - 规避：加超时与 backoff；超时写 trace：`stage="timeout_*"` 并触发 STOP_FLAG

2) **实现阶段对齐失败（n_test 不一致）**
   - 现状：测试从 499 倒跑到 350，且跳级窗口可能导致“没跑表却 table_number 变化”
   - 规避：沿用 `Dynamic_ALNS_RL34959.py` 中的 implement_start_synced 逻辑，并在 baseline replay 强制使用 RL 的 table 序列

3) **日志字段不一致导致 CSV 结构损坏**
   - 关键点：`rl_trace.csv` 同时被 RL 与 ALNS 写入，必须共用同一套 fieldnames
   - 规避：本次对 `dynamic_RL34959.TRACE_FIELDS` 与 `Intermodal_ALNS34959.TRACE_FIELDS` 同步增量字段，保持一致

4) **收敛误判导致过早跳级**
   - 现状：rolling_avg 基于最近 rewards，且 reward 可能稀疏/噪声大
   - 规避：把收敛判据从“纯 reward 阈值”升级为 “漂移分段内的稳态 + 置信区间 + 最小样本量”

---

## 附录（系统设定与关键实现摘录）

### A1) 动作空间
- `codes/dynamic_RL34959.py:coordinationEnv.__init__()`：`self.action_space = Discrete(2)`

### A2) 单步 episode（ALNS 模式）
- `codes/dynamic_RL34959.py:main()`：`episode_length = 1`
- `codes/dynamic_RL34959.py:coordinationEnv.step()`：`done = (self.horizon_length == episode_length)`

### A3) 共享表同步（核心耦合点）
- RL 写入动作：`codes/dynamic_RL34959.py:send_action()`
- RL 轮询 reward：`codes/dynamic_RL34959.py:coordinationEnv.step()`
- ALNS 初始化表：`codes/Intermodal_ALNS34959.py:real_main()`（`state_reward_pairs = pd.DataFrame(...)`）

### A4) 漂移信号（可得但未进入 obs）
- 生成器写入：`codes/generate_mixed_parallel.py` → Excel `__meta__`（`gt_mean/phase_label`）
- RL 读取：`codes/dynamic_RL34959.py:get_state()` → `current_gt_mean/current_phase_label`
- trace 记录：`codes/dynamic_RL34959.py:log_trace_from_row()` 与 `codes/Intermodal_ALNS34959.py:log_rl_event()`

