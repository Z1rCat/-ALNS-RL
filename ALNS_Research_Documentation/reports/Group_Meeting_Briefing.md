# 组会汇报讲稿草稿：分布漂移可视化 + 课程式跳级（ALNS‑RL）

> 说明：本文只写“工作逻辑与实现”，**结果展示/图表解读请你自行补充**（我已预留空白段落）。

---

## 1. 上次组会的任务背景：把“分布变换”展示在 RL Reward 曲线里

我们的系统是 **ALNS‑RL 协同的多式联运动态优化**：

- **ALNS**：负责仿真与路径/调度优化（每个实例一个 Excel 环境文件）。
- **RL（DQN/A2C/PPO 等）**：在发生不确定事件（拥堵/延误）时，做二分类决策：
  - `action=0`：等待/保持（Keep/Wait）
  - `action=1`：重规划（Reroute）
- **智能跳级（Curriculum Jumping）**：RL 达到某个收敛条件后，不必按 `0..499` 顺序跑完全部文件，而是根据场景结构跳到下一个阶段或直接进入测试。
- **物理隔离（run_dir）**：每次运行输出到独立目录 `codes/logs/run_*/`，包含 `data/`、日志 CSV、meta 配置等，便于复现实验与对比。

上次组会老师提出：在最终绘制 RL reward 曲线时，要把**分布切换**也明确展示出来，否则 reward 曲线缺乏“环境背景”。

对应实现落在 `codes/plot_paper_figure.py`（Fig2 右轴 + 背景分段），以及日志中对环境标签的记录（见第 3 节）。

---

## 2. 实现可视化后发现的漏洞：RL 可能“过早收敛”，导致阶段没跑全

在把分布切换画进 reward 曲线后，我发现了一个关键漏洞：

- 现有“智能跳级”只看 **RL 是否收敛**（rolling average reward 是否达标），一旦达标就可能 **直接进入测试阶段**。
- 但对于**多阶段场景**（例如 A→B、A→B→A），如果 RL 在阶段 A 很快达标，就会出现：
  - 训练阶段还没充分经历阶段 B（甚至没进入 B）
  - 就直接切到测试，导致论文/报告里对“适应/记忆/风险”能力的论证不严谨

因此，我把“跳级”从“仅由收敛触发”升级为“由 **场景结构（多阶段） + 收敛** 联合决定”：

- 多阶段场景：先跳到下一阶段，直到关键阶段都达标，再进入测试。
- 单阶段训练 + 测试 OOD 的场景：允许训练阶段只学习 A，测试阶段再看 B（这本身就是设计目标）。

---

## 3. 关键改动 1：生成数据时打标签，并把标签写进日志

### 3.1 训练/测试划分（500 个文件）

我采用了一个清晰的划分方式，且与代码的“倒序测试”机制一致：

- 一次实验生成 500 个环境文件（`table_number = 0..499`）。
- **训练阶段**使用前 350 个（概念上对应 `0..349`）。
- **测试阶段**使用后 150 个（概念上对应 `350..499`）。

同时，系统内部实现上，测试阶段是“从末尾往回跑”（例如从 499 递减到 350），这是历史代码已有的实现方式。

补充说明：在近期调试中我也遇到过一个实现层面的 bug——当 `implement=1` 切换发生时，如果 ALNS 线程继续“自增/自减 table_number”但实际并未运行对应实例，会出现“看似进入测试，但立刻到达边界 350”的假象。为保证测试段确实从 499 开始跑，我补了一个 **implement 切换时的 table_number 同步与等待机制**（见 `codes/Dynamic_ALNS_RL34959.py` 的 `implement_start_synced` 相关逻辑）。

### 3.2 标签如何产生

每个 Excel 环境文件都会带一个 `__meta__` sheet，里面至少包含：

- `gt_mean`：该实例所属分布的均值（或用作环境难度 proxy）
- `phase_label`：阶段标签（例如 `A/B/C`）

RL 在构建状态时会读取该 sheet，并把标签写入日志：

- 读取位置：`codes/dynamic_RL34959.py` 的 `get_state()`（读取 `__meta__`，更新 `current_gt_mean/current_phase_label`）
- 写入日志：`rl_trace.csv` 中的 `gt_mean`、`phase_label` 列

因此可视化脚本可以直接从日志中拿到“环境处于哪个阶段/均值是多少”。

---

## 4. 关键改动 2：六类分布设计（验证六类能力）

我把实验场景抽象为六类“代表性分布机制”，每类用于验证 RL 的不同能力假设。

这些场景在 `distribution_config.json` 中有明确的 `pattern/means/variance` 描述，并在批量脚本中作为 Top‑6 运行集合使用：

| 类别 | 代表场景 | pattern | 设计理念（验证什么） | 训练/测试结构要点 |
|---|---|---|---|---|
| 基准（混合） | `S1_1` | `random_mix` | 环境以混合方式随机出现：检验“鲁棒决策”与抗噪能力 | 训练/测试都处于混合机制 |
| 适应（均值突变） | `S5_1` | `adaptation` | A→B 后长期处于高压：检验“快速适应 + 稳定策略” | 训练含 A 与部分 B；测试继续 B |
| 记忆（ABA 回归） | `S2_1` | `aba` | A→B→A：检验“灾难性遗忘/记忆恢复” | 训练覆盖 A+B；测试回到 A |
| 泛化（OOD） | `S3_1` | `ab` | 训练只见 A，测试给出 B：检验“分布外泛化” | `ab` 刚好使 A≈350、B≈150，与 350/150 切分对齐 |
| 演变（ABC） | `S6_1` | `abc` | A→B→C：检验“非平稳长期演化”下策略保持/迁移 | 训练覆盖 A+B；测试侧重 C |
| 风险（方差突变） | `V1_3` | `aba`（均值恒定） | 均值不变、方差突变再恢复：检验“风险敏感/不确定性方差适应” | 训练含低方差 + 高方差；测试回到低方差 |

> 注：这里的关键点是把 500 个实例的分段比例与 `0..349 / 350..499` 的训练‑测试切分对齐，让每类场景的“测试目标”清晰可解释。

---

## 5. 关键改动 3：每类分布都有对应的“收敛 + 跳转”逻辑

### 5.1 RL 侧收敛判定（用于跳级）

RL 会周期性做 evaluation，并维护一个滑动窗口平均：

- 滑动窗口长度：`SLIDING_WINDOW = 30`
- 收敛阈值：`CURRICULUM_REWARD_THRESHOLD = 0.7`
- 连续满足次数：`CURRICULUM_SUCCESS_REQUIRED = 3`
- 触发标志：`curriculum_converged = 1`

对应代码在 `codes/dynamic_RL34959.py`：

- `rolling_avg >= threshold` 时 `sucess_times += 1`，否则清零
- 连续 3 次达标才认为 curriculum 收敛（用于跳级）
- 为了本地快速 Debug，我把 `S0_Debug` 的阈值降到了 `0.3`

重要说明：这里的 `curriculum_converged` **不是“最终最优收敛”**，它是“用于跳级的阶段性达标信号”，目的是把算力集中到更难/更关键的阶段或测试段。

### 5.2 ALNS‑RL 协同侧的跳级控制（按场景 pattern 决策）

跳级控制在 `codes/Dynamic_ALNS_RL34959.py`：

- ALNS 线程（`approach=0`）维护 `table_number` 作为当前实例索引
- 一旦 `dynamic_RL34959.curriculum_converged == 1`，就根据 `distribution_config.json` 里该场景的 `pattern` 决定下一步：
  - 多阶段：跳到下一阶段边界（例如 S5 从 A 达标后跳到 B 的起点）
  - 设计为 OOD 的场景（S3）：训练阶段只跑 A，达标后直接进入测试（B）
  - 记忆/风险（ABA）：训练完成 A+B 后进入测试（末段 A），验证“回归与恢复”

---

## 6. 举例：两类场景的跳转逻辑（方便组会口头解释）

### 6.1 适应场景 S5（A→B，训练中强制进入 B 再测试）

目标：避免 RL 在 A 很快学会后直接去测试，导致“适应性”论证不成立。

逻辑（`codes/Dynamic_ALNS_RL34959.py`）：

1. 训练阶段在 A 中达标（`curriculum_converged=1`）
2. 不进入测试，而是把 `table_number` 跳到 B 的起点（例如 `100`）
3. 在 B 中继续训练，满足最小 B 训练量后再允许进入测试（例如 `min_phase_b_train=105`）
4. 进入测试：把 `implement=1`，并从 `table_number=499` 开始倒序测试

### 6.2 风险场景 V1（均值恒定、方差突变）

难点：均值不变时，单画 `gt_mean` 无法体现环境切换。

因此在 Fig2 我做了一个“自动切换指标”的规则（见第 8 节）：

- 如果检测到该分布存在方差变化且均值恒定，则右轴用 `gt_std`（标准差）替换 `gt_mean`；
- 并在图中标注“均值恒定为 X”，避免审稿/老师质疑。

---

## 7. 基准策略回放：Always‑Wait / Always‑Reroute / Random（可选）

为了做严谨对比，我加入了**基准策略回放脚本** `codes/run_benchmark_replay.py`，核心目标：

- 与 RL **复用同一套环境文件**（`run_dir/data/`），不允许重新生成数据
- 严格对齐 RL 实际跑过的 `table_number` 序列（智能跳级后序列是动态的）
- 不加载 RL 模型，直接固定 action：
  - Always‑Wait：`action=0`
  - Always‑Reroute：`action=1`
  - Random：`action∈{0,1}`（默认不跑，`--include-random` 才加入）

实现要点：

- 轨迹对齐：从 `rl_trace.csv` 解析 `(table_number, phase)` 序列
- 阶段一致性：每一步根据历史 phase 设置 `dynamic_RL34959.implement`
- “闭环等待”：发出 action 后轮询 ALNS 的共享标志位，直到 reward 回填成功
- 输出：在同一 run 目录下生成 `baseline_wait.csv / baseline_reroute.csv / baseline_random.csv`

对比目的（口头表述建议）：

> “我们不仅要证明 RL 比弱基准（Always‑Wait）强，还要证明它能在高压拥堵下对抗强基准（Always‑Reroute）；并通过 Random baseline 排除‘偶然性’。”

---

## 8. 可视化脚本升级：支持“均值漂移 + 方差漂移”，并解决图例遮挡

主脚本：`codes/plot_paper_figure.py`（输出到 `run_dir/paper_figures/`）。

### 8.1 Fig2 右轴环境线：gt_mean vs gt_std 的切换逻辑

当前规则：

1. **优先判断方差是否变化**（从 `distribution_config.json` 读取 `variance`，并基于 `phase_label` 映射出 `gt_std = sqrt(variance)`）
2. 如果“方差变化 & 均值恒定（gt_mean 不变）”，则：
   - 右轴画 `gt_std`（环境标准差），替代 `gt_mean`
   - 图内注释框写：`均值恒定: X.X`
3. 其他情况：沿用原逻辑画 `gt_mean` 阶梯线

这让 V1 类“变方差”场景在适应曲线中也能直观看到环境切换。

### 8.2 Fig2 图例遮挡：移动到图外上方

为避免图例遮挡右上角关键曲线，我把双轴 legend 合并后移到图外上方：

- 位于标题下方、绘图区上方
- 水平排列（`ncol>=3`）
- 去边框（`frameon=False`）

---

## 9. 本地批量实验与多核并行：为服务器实验做“最小可跑样例”

为解决“多分布 × 多算法 × 多 R × 多基准”组合带来的时间压力，我写了批量脚本：

- 公共调度：`codes/run_experiments_common.py`
- 本地调试：`codes/run_experiments_local.py`
- 服务器全量：`codes/run_experiments_server.py`

每个任务的闭环流程：

1. 运行主程序：`codes/Dynamic_master34959.py`（生成数据 + ALNS‑RL 训练/测试 + 日志落盘）
2. 跑 baseline：`codes/run_benchmark_replay.py --policy all`
3. 画图：`codes/plot_paper_figure.py`

并发策略（本地）：

- 以“独立 run_dir”为最小并发单元（降低写冲突）
- 通过 `ThreadPoolExecutor` 并发启动多个 Python 子进程
- 默认并发数建议：`物理核数 - 2`，本地调试先用 2~3 个并发稳定验证
- 数据生成阶段默认单核（`generator_workers=1`），避免并发写 Excel 引发损坏（历史确实出现过 `BadZipFile/EOFError`）

（本地测试结果展示：待补充）

---

## 10. 下一步计划：Docker + 服务器全量实验（请老师给建议）

下一步我计划：

1. **Docker 化部署**：固定 Python/依赖版本、字体与绘图后端，保证服务器可复现实验与出图。
2. **服务器全量实验**：Top‑6 分布 × `R∈{5,30}` × `{DQN,PPO,A2C}` × `{42,123,2024}`，并生成 `Mean±Std` 的对比图与汇总表。

我想请教老师的几个问题（组会可直接问）：

- 对于 `S3_1` 这种 **train=A / test=B** 的 OOD 设计，老师更希望“完全不在训练见到 B”，还是允许少量 curriculum 进入 B？
- 论文里基准策略是否只保留 `Always‑Wait` 和 `Always‑Reroute` 即可？`Random` 是否需要作为第三个基准？
- 服务器实验的资源分配上，更推荐“并发跑多个 run”（多进程）还是“单 run 内部更深度并行”（例如 ALNS 内部并行）？

（服务器实验结果展示：待补充）

---

## 结果展示（你来补充）

- 关键图：Fig1–Fig4（每个场景至少 1 组）
- 本地 Debug 组合：`S5_1 / S3_1 / V1_3`，`R=30`，`seed=42`，算法：`DQN/A2C`
- 你想重点讲的结论：______
