# ALNS–RL 动态重规划实验：日志与结果分析汇报

> 本报告基于自动生成的运行日志与图片（`codes/logs/run_*`），用于给导师/组会汇报。  
> 重点回答：**RL 在什么时候被调用、两个阶段动作含义如何区分、reward 如何解释、不同分布与请求规模 R 下效果如何**。

---

## 1. 研究背景与目标

动态多式联运场景中存在突发不确定事件（如拥堵）。本项目采用三层架构：

- **底层优化层（ALNS）**：`codes/Intermodal_ALNS34959.py`，负责路径搜索、可行性检测、时间轴推进与事件模拟。
- **智能体层（RL）**：`codes/dynamic_RL34959.py`，基于 Stable-Baselines3（DQN/PPO），封装 Gym 环境并输出离散动作。
- **协调层（入口/线程管理）**：`codes/Dynamic_ALNS_RL34959.py` 与 `codes/Dynamic_master34959.py`，负责数据生成、启动双线程与同步共享变量。

目标是让 RL 在关键时刻给出“是否需要重规划/是否接受插入”的决策，提高动态过程中的可行性与稳定性。

---

## 2. 系统交互流程：ALNS 线程 ↔ RL 线程

当 ALNS 遇到突发事件（拥堵 begin/finish 等）时，会在共享内存（Pandas DataFrame，例如 `state_reward_pairs`）写入一条“待决策状态”；RL 侧读取状态并写回动作；随后 ALNS 执行动作并计算奖励，再写回 reward。

在日志中，RL 的**调用时刻**主要体现在 `rl_trace.csv` 的 `stage` 字段：

- `begin_removal`：进入**规划/破坏阶段**，需要 RL 决定 `Wait(0)` 或 `Reroute(1)`。
- `begin_insertion`：进入**插入/修复阶段**，需要 RL 决定 `Accept(0)` 或 `Reject(1)`。
- `send_action`：RL 已发送动作。
- `receive_reward`：RL 收到 reward（来自 ALNS 可行性判断/反馈）。

---

## 3. 动作语义与奖励定义（报告解读核心）

### 3.1 两个阶段共用动作 0/1，但语义不同

同一个动作编号在不同阶段含义完全不同，日志与绘图已做区分（分别生成 Removal/Insertion 的动作分布图）：

- **规划/破坏阶段（Removal）**
  - `action=0`：等待/保持原路（Wait）
  - `action=1`：移除受影响订单/触发重规划（Reroute）
- **插入/修复阶段（Insertion）**
  - `action=0`：接受贪婪插入的新路径（Accept）
  - `action=1`：拒绝插入（Reject）

### 3.2 reward 的含义：本质是“可行性二分类正确率”

在 `codes/Intermodal_ALNS34959.py` 的 `get_reward()` 中，reward 采用 **0/1** 的离散规则：

- 若 ALNS 判定“当前方案可行（`bool_or_route` 不是 bool）”，则 **正确动作应为 `0`**（保持/接受），reward=1，否则 reward=0。
- 若 ALNS 判定“当前方案不可行（`bool_or_route` 是 bool）”，则 **正确动作应为 `1`**（重规划/拒绝），reward=1，否则 reward=0。

因此：

- `reward=1` ≈ 本次决策与“可行性标签”一致（分类正确）
- `reward=0` ≈ 分类错误
- **实施阶段平均 reward** 可直接解释为 **“决策正确率（准确率）”**

---

## 4. 日志数据结构（用于绘图与复现）

每次运行对应一个文件夹：`codes/logs/run_YYYYMMDD_HHMMSS_R{R}_{dist}/`，包含：

- `meta.json`：本次运行配置
  - `distribution`：分布名称（dist）
  - `request_number`：请求规模（R）
- `rl_trace.csv`：逐事件“交互流水”（最细粒度）
  - 关键字段：`ts, phase, stage, uncertainty_index, request, vehicle, delay_tolerance, severity, action, reward, action_meaning, source`
  - `phase`：`train/eval/implement`
  - `stage`：`begin_removal/send_action/receive_reward/begin_insertion/...`
  - `action_meaning`：已映射为中文含义（如“等待/保持”“重新规划”“接受插入”“拒绝插入”）
- `rl_training.csv`：逐步训练/评估/实施的日志（用于曲线/直方图）
  - 主要字段：`ts, phase, step_idx, reward, avg_reward, std_reward, training_time, implementation_time`
  - 其中 `phase=implement` 的 `reward` 用于实施阶段 reward 直方图与曲线
- `rl_summary.csv`：实施阶段汇总统计（用于跨 run 对比）
  - `reward_count, average_reward, std_reward`
  - `removal_action/removal_wait_action` 与各自累计 reward
  - `insertion_action/insertion_non_action` 与各自累计 reward
- 自动生成的图片（每个 run 一套）
  - `train_reward_curve.png`、`eval_reward_curve.png`、`implement_reward_curve.png`
  - `reward_hist.png`
  - `action_dist_removal.png`、`action_dist_insertion.png`
  - `stage_counts.png`

---

## 5. 我们做了什么（数据汇总与图形重现）

为确保分析可复现、可以持续补充更多 run，本次工作新增了两段辅助脚本，统一收集日志并生成跨运行对比图：

- `ALNS_Research_Documentation/collect_rl_logs.py`：遍历 `codes/logs/run_*`，提取 `meta.json`、`rl_summary.csv`、`rl_training.csv` 中的关键字段并按 `(R, dist)` 聚合，输出 `ALNS_Research_Documentation/rl_logs_aggregate.csv`，方便做表格与热力图。
- `ALNS_Research_Documentation/plot_rl_logs_summary.py`：读取上一步生成的聚合表，绘制两张图并保存到 `ALNS_Research_Documentation/figures_rl_logs/`：
  1. 实施阶段平均 reward 热力图（同一组图中横轴 R，纵轴 dist）；
  2. 规划/插入阶段动作比例与平均 reward 的散点图（展示动作塌缩与效果的关系）。

追加的两张图片也被引用在本报告中（第6节）。

运行方式示例（已在附录记录）：

```powershell
& A:/MYpython/34959_RL/codes/env/python.exe a:/MYpython/34959_RL/ALNS_Research_Documentation/collect_rl_logs.py
& A:/MYpython/34959_RL/codes/env/python.exe a:/MYpython/34959_RL/ALNS_Research_Documentation/plot_rl_logs_summary.py
```

---

---

## 6. 不确定性分布与生成规则

数据生成逻辑位于 `codes/generate_mixed_parallel.py` 的 `get_distribution_matrix()`。

**关键点：混合分布不是“逐样本随机混合”，而是“按实例编号分段拼接”。**

- 例如 `normal_mix_8_80_75_25`：前 75% 实例来自均值 8 的正态分布，后 25% 来自均值 80 的正态分布（先低后高）。
- 例如 `lognormal_mix_80_8_25_75`：前 25% 实例来自均值更大的对数正态（更“高压”），后 75% 来自更“低压”的对数正态（先高后低）。
- 三段混合 `lognormal_mix_9_30_3_30_30_40`：前 30%（均值 9）+ 中 30%（均值 30）+ 后 40%（均值 3）。

这意味着：如果训练/实施在时间上更偏向某一段实例区间，可能出现“分布漂移”，进而影响策略泛化。

---

## 7. 总体结果概览（跨 run 聚合）

为便于汇报，已将所有 `run_*` 汇总为配置级表格：`ALNS_Research_Documentation/rl_logs_aggregate.csv`（按 `(R, dist)` 聚合）。

同时新增两张跨 run 的对比图（存放于 `ALNS_Research_Documentation/figures_rl_logs/`）。

### 6.1 不同 R 与分布下：实施阶段平均奖励（=正确率）

![](figures_rl_logs/overall_implement_avg_reward_heatmap.png)

读图说明：

- 横轴：请求规模 `R`
- 纵轴：分布 `dist`
- 单元格数值：实施阶段平均 reward（0~1，越大越好）

### 6.2 “动作塌缩”与效果的关系（配置级散点）

![](figures_rl_logs/overall_reward_vs_action_ratio.png)

读图说明：

- 左图横轴：规划阶段 `Reroute(1)` 的占比；右图横轴：插入阶段 `Accept(0)` 的占比
- 纵轴：实施阶段平均 reward（正确率）
- 不同颜色表示不同的 `R`

经验现象（对组会讲述很有用）：

- 某些配置会出现“几乎总是 Reroute / 几乎总是 Reject”的**极端动作偏好**（动作塌缩）。
- 动作塌缩不一定坏：当真实标签分布极不均衡时，塌缩策略也可能得到很高正确率；但在标签接近均衡时，塌缩会导致正确率接近 0.5（近似随机）。

### 6.3 配置汇总表（按配置聚合）

说明：

- `实施平均奖励`：实施阶段的平均 reward（≈正确率）
- `规划阶段Wait/Reroute次数`、`插入阶段Accept/Reject次数`：来自 `rl_summary.csv` 的统计（若同配置有多次 run 则取均值）
- `训练步数(均值)`：来自 `rl_training.csv` 中 `phase=train` 的 reward 记录行数均值

| R | 分布(dist) | 重复次数 | 实施平均奖励 | 实施奖励标准差 | 训练步数(均值) | 规划阶段Wait次数 | 规划阶段Reroute次数 | 插入阶段Accept次数 | 插入阶段Reject次数 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | lognormal_mix_8_80_50_50 | 1 | 0.975 |  | 100.0 | 190.0 | 5.0 | 5.0 | 0.0 |
| 5 | normal_mix_8_80_50_50 | 1 | 0.965 |  | 120.0 | 188.0 | 6.0 | 6.0 | 0.0 |
| 5 | lognormal_mix_9_30_3_30_30_40 | 1 | 0.895 |  | 100.0 | 194.0 | 4.0 | 2.0 | 0.0 |
| 5 | normal_mix_80_8_75_25 | 1 | 0.680 |  | 100.0 | 197.0 | 2.0 | 1.0 | 0.0 |
| 5 | normal_mix_8_80_75_25 | 4 | 0.339 | 0.418 | 121.0 | 61.8 | 73.5 | 4.0 | 60.8 |
| 5 | lognormal_mix_8_80_75_25 | 1 | 0.060 |  | 100.0 | 3.0 | 97.0 | 2.0 | 98.0 |
| 20 | normal_mix_8_80_75_25 | 1 | 0.965 |  | 100.0 | 187.0 | 7.0 | 6.0 | 0.0 |
| 20 | lognormal_mix_8_80_75_25 | 1 | 0.940 |  | 100.0 | 185.0 | 5.0 | 9.0 | 1.0 |
| 20 | normal_mix_80_8_75_25 | 1 | 0.545 |  | 136.0 | 197.0 | 2.0 | 1.0 | 0.0 |
| 20 | lognormal_mix_9_30_3_30_30_40 | 2 | 0.508 | 0.555 | 102.0 | 97.5 | 49.5 | 8.5 | 44.5 |
| 30 | lognormal_mix_80_8_25_75 | 1 | 0.980 |  | 100.0 | 193.0 | 4.0 | 3.0 | 0.0 |
| 30 | lognormal_mix_8_80_50_50 | 1 | 0.975 |  | 100.0 | 189.0 | 5.0 | 6.0 | 0.0 |
| 30 | normal_mix_80_8_25_75 | 1 | 0.960 |  | 144.0 | 183.0 | 7.0 | 9.0 | 1.0 |
| 30 | normal_mix_8_80_50_50 | 2 | 0.960 | 0.021 | 100.0 | 189.0 | 6.5 | 4.5 | 0.0 |
| 30 | normal_mix_8_80_75_25 | 3 | 0.655 | 0.524 | 157.3 | 125.7 | 35.7 | 4.7 | 34.0 |
| 30 | lognormal_mix_9_30_3_30_30_40 | 3 | 0.635 | 0.468 | 128.0 | 124.3 | 34.7 | 8.3 | 32.7 |
| 30 | lognormal_mix_8_80_75_25 | 2 | 0.510 | 0.665 | 110.0 | 100.0 | 45.0 | 6.5 | 48.5 |
| 30 | lognormal_mix_80_8_50_50 | 1 | 0.490 |  | 100.0 | 2.0 | 113.0 | 1.0 | 84.0 |
| 30 | normal_mix_80_8_75_25 | 2 | 0.443 | 0.124 | 116.0 | 98.5 | 48.5 | 4.5 | 48.5 |
| 50 | normal_mix_8_80_75_25 | 1 | （运行异常/无有效统计） |  |  |  |  |  |  |

---

## 8. 典型案例解读（从图到机制）

下面选择三个代表性配置，用图片解释“RL 在哪个阶段被调用、学到了什么/塌缩成什么”。

### 7.1 失败案例：R=5，lognormal_mix_8_80_75_25（正确率 0.06）

关键统计（`rl_summary.csv`）：

- 实施平均奖励：0.06（200 次交互仅约 6% 正确）
- 规划阶段：`Reroute(1)` 97 次且 reward 全为 0（几乎全部错）
- 插入阶段：`Reject(1)` 98 次但仅 8 次正确

图 1：实施阶段 reward 直方图（几乎全是 0）

![](../codes/logs/run_20251218_121215_R5_lognormal_mix_8_80_75_25/reward_hist.png)

图 2：规划阶段动作分布（实施阶段几乎总是 Reroute）

![](../codes/logs/run_20251218_121215_R5_lognormal_mix_8_80_75_25/action_dist_removal.png)

图 3：插入阶段动作分布（实施阶段几乎总是 Reject）

![](../codes/logs/run_20251218_121215_R5_lognormal_mix_8_80_75_25/action_dist_insertion.png)

解释（可用于汇报口径）：

- 策略出现“**双阶段同时塌缩到更激进动作**”（规划阶段几乎总重规划，插入阶段几乎总拒绝）。
- 但该配置下可行性标签显然更偏向“可行”（否则不会出现大量 reward=0），因此塌缩策略几乎全错。

### 7.2 成功案例：R=30，lognormal_mix_80_8_25_75（正确率 0.98）

关键统计（`rl_summary.csv`）：

- 实施平均奖励：0.98（接近满分）
- 规划阶段：`Wait(0)` 193 次且全部正确
- 插入阶段：`Accept(0)` 3 次且全部正确

图 1：实施阶段 reward 直方图（几乎全是 1）

![](../codes/logs/run_20251219_144532_R30_lognormal_mix_80_8_25_75/reward_hist.png)

图 2：规划阶段动作分布（实施阶段几乎总是 Wait）

![](../codes/logs/run_20251219_144532_R30_lognormal_mix_80_8_25_75/action_dist_removal.png)

图 3：插入阶段动作分布（实施阶段基本全 Accept）

![](../codes/logs/run_20251219_144532_R30_lognormal_mix_80_8_25_75/action_dist_insertion.png)

解释：

- 在该配置下，“保持/接受”更符合可行性标签分布；策略选择相对保守，因此正确率非常高。
- 值得进一步确认：这是真正学到了状态-动作映射，还是“标签分布极不均衡导致的合理塌缩”（后续可通过更均衡的测试集验证）。

### 7.3 中等案例：R=30，lognormal_mix_80_8_50_50（正确率 0.49）

关键统计（`rl_summary.csv`）：

- 实施平均奖励：0.49（接近随机猜测）
- 规划阶段：`Reroute(1)` 113 次，其中 59 次正确（约 52%）
- 插入阶段：`Reject(1)` 84 次，其中 37 次正确（约 44%）

图 1：实施阶段 reward 直方图（0/1 接近对半）

![](../codes/logs/run_20251219_171452_R30_lognormal_mix_80_8_50_50/reward_hist.png)

图 2：规划阶段动作分布（实施阶段偏向 Reroute，但并不总对）

![](../codes/logs/run_20251219_171452_R30_lognormal_mix_80_8_50_50/action_dist_removal.png)

图 3：插入阶段动作分布（实施阶段偏向 Reject，但并不总对）

![](../codes/logs/run_20251219_171452_R30_lognormal_mix_80_8_50_50/action_dist_insertion.png)

解释：

- 当标签更接近均衡时，“动作塌缩”会直接把正确率拉回 0.5 左右。
- 这类配置更能暴露“状态信息不足/策略泛化不足”的问题，是后续优化的重点测试集。

---

## 9. 讨论：主要现象与可能原因

结合跨 run 统计与典型案例，当前系统呈现出以下规律（适合作为组会讨论点）：

1. **reward=0/1 是可行性分类正确率**：平均 reward 高未必代表成本更优，只能说明“决策与可行性标签一致”。
2. **动作塌缩很常见**：不少配置中，实施阶段几乎总选某个动作（例如总 Reroute 或总 Wait）。
3. **分布分段拼接可能带来漂移**：混合分布按“前段/后段”生成，若训练与实施在实例编号上偏向不同区间，可能导致策略不稳。
4. **状态维度较少**：当前状态主要围绕 `delay_tolerance`、`severity`（以及可选的事件类型）构成；对复杂约束来说，信息可能不足以稳定区分可行/不可行。
5. **大 R（如 R=50）仍存在已知异常**：目前已出现数组拼接维度不一致等错误，导致该规模未被纳入稳定统计。

---

## 10. 结论与下一步建议（面向导师的“可执行”清单）

1. **提升评测的可解释性**：在报告中将“正确率（reward）”与“成本改进/可行性保持”区分，必要时引入基于 cost gap 的连续奖励做对照实验。
2. **降低动作复用歧义**：将 Removal/Insertion 拆成两个 policy head（或两个独立策略），避免“同编号动作跨阶段语义冲突”带来的学习困难。
3. **改进混合分布生成**：在保持总体比例的前提下，对实例顺序进行随机打散（shuffle），降低训练/实施的分布漂移。
4. **扩展状态特征**：加入更直接的可行性相关特征（如关键时间窗 slack、受影响段长度、当前路径剩余缓冲、是否经过关键枢纽等）。
5. **补齐大规模 R 的健壮性**：优先修复 R=50 的维度不一致问题，再做规模扩展实验（R=50/100）。

---

## 附录：如何复现实验与再出图

建议使用项目虚拟环境运行（PowerShell）：

```powershell
& A:/MYpython/34959_RL/codes/env/python.exe a:/MYpython/34959_RL/codes/Dynamic_master34959.py
& A:/MYpython/34959_RL/codes/env/python.exe a:/MYpython/34959_RL/codes/tools/plot_rl_logs.py
```

聚合统计表与跨 run 对比图：

- `ALNS_Research_Documentation/rl_logs_aggregate.csv`
- `ALNS_Research_Documentation/figures_rl_logs/overall_implement_avg_reward_heatmap.png`
- `ALNS_Research_Documentation/figures_rl_logs/overall_reward_vs_action_ratio.png`

