# 项目事实报告

日期：2026-03-07  
用途：向外部强化学习领域研究者客观介绍本项目的目标、代码结构、实验流程、已测试算法、真实结果与已观测到的问题。  
说明：本报告只基于项目中的真实代码、配置、日志、结果文件整理，不包含算法推荐，不包含主观设计建议。

---

## 1. 项目目标

本项目是一个面向 **动态 ALNS 决策** 的强化学习研究工程，核心任务是在动态干扰/分布变化条件下，让 RL 决策模块在不同事件分布上学习 removal / insertion 相关决策，并评估其：

- OOD（out-of-distribution）泛化能力
- forgetting / generalization / random mix 等时序分布切换下的表现
- 在动态 ALNS 决策过程中的适应性与动作坍缩问题

从代码与分布配置看，当前项目显式组织了多类测试分布：

- `O_*`：OOD 分布
- `F1_*` / `F2_*`：forgetting 分布
- `G_*`：generalization 分布
- `R_*`：random_mix 分布
- 还有若干历史日志中的单独分布，如 `S3_1`

相关配置入口：

- `codes/Dynamic_master34959.py`
- `distribution_config.json`

---

## 2. 代码与工程整体结构

### 2.1 主入口与总体调度

主入口文件：

- `codes/Dynamic_master34959.py`

该文件负责：

- 选择分布与请求规模
- 选择算法
- 调用训练、评估、outer pipeline
- 组织 run 目录与环境变量

### 2.2 环境与训练主逻辑

环境、训练与评估主逻辑位于：

- `codes/core/dynamic_RL34959.py`

该文件负责：

- 构造 RL 环境
- 调用 SB3 或项目自定义 agent
- 管理 `train_only / eval_only / train_eval`
- 记录 `rl_trace.csv / rl_training.csv / rl_summary.csv`
- 加载与保存 checkpoint

### 2.3 外层分布选择与 outer RL / replay 管线

outer pipeline 相关文件：

- `codes/outer_rl/run_edrl_phase2.py`
- `codes/outer_rl/run_edrl_pipeline.py`

它们负责：

- outer candidate action space
- phase2 / phase3 迭代
- EDRL / RARL / PLR_UED / SABER 类 outer 逻辑
- replay buffer、bandit / TS / UCB / DQN outer policy
- phase4 checkpoint 选择与 implement 评估

### 2.4 实验完成标记与结果完整性判定

实验整包完成逻辑位于：

- `codes/experiments/run_experiments_common.py`

该模块定义了：

- `DONE.json`
- heartbeat / lock
- baseline / metrics / plot 等后处理是否完成

因此：

- 有 `DONE.json`：可视为整包完成
- 无 `DONE.json` 但 `rl_summary.csv + rl_trace.csv` 存在：主结果可用，但整包未完成
- 缺少 `rl_summary.csv`：不能作为最终结果使用

### 2.5 主要算法实现位置

| 算法族 | 主要代码位置 | 说明 |
|---|---|---|
| PPO / A2C / 标准 SB3 算法 | `codes/core/dynamic_RL34959.py` | 通过 SB3 调用 |
| PPO_NEW v3/v3.1 | `codes/algorithms/ppo_new/nn/context_extractor.py` 及相关 `ppo_new` 目录 | 三路分支编码 + 聚合 |
| CADM | `codes/robust_rl/ppo_new/v6_cadm.py` | 在 PPO_NEW 主干上增加辅助预测头 |
| CQL_DQN | `codes/robust_rl/cql_dqn.py` | DQN + conservative regularization |
| EDRL / NOVA_EDRL / RARL / PLR_UED / SABER | `codes/outer_rl/run_edrl_phase2.py` + `codes/outer_rl/run_edrl_pipeline.py` | outer 选择与四阶段流程 |

---

## 3. 训练 / 测试流程

当前工程的典型运行流程可以概括为四段：

1. **Phase1**
   使用某个基础分布训练 inner policy，并保存 `theta_phase1.zip`

2. **Phase2 / Phase3**
   outer 管线生成 candidate 分布，运行短训/短评估，记录 outer objective 与 replay 统计

3. **Phase4**
   选择一个 outer checkpoint，执行 `eval_only` implement 评估

4. **后处理**
   写汇总、baseline、图表与 `DONE.json`

当前项目中，`phase4` checkpoint 选择逻辑已在 `run_edrl_pipeline.py` 中实现多个策略：

- `latest_phase3`
- `best_phase3_objective`
- `best_any_objective`
- `latest_any`

---

## 4. 日志与结果文件结构

典型 run 目录中常见文件：

- `meta.json`
- `rl_summary.csv`
- `rl_trace.csv`
- `rl_training.csv`
- `rl_decision.csv`（部分较新 run 才稳定有）
- `console_output.txt`
- `post_stage/outer_train_round.csv`
- `post_stage/outer_actions.csv`
- `post_stage/outer_plr_stats.csv`
- `post_stage/checkpoints/*.zip`
- `DONE.json`

其中：

- `rl_summary.csv`：最终 implement 汇总
- `rl_trace.csv`：更细粒度状态-动作-回报记录
- `rl_decision.csv`：较新的决策级日志，适合 trace-level 切片
- `outer_train_round.csv`：outer 每轮 objective 与 candidate 统计
- `outer_actions.csv`：outer 选择的 candidate level 详情
- `outer_plr_stats.csv`：replay / new sample 比例等

---

## 5. 结果来源与可信度分类

本报告使用的结果分三类：

### 5.1 完整可信（优先）

判定条件：

- `DONE.json` 存在
- `rl_summary.csv` 存在

本类主要来自：

- `codes/nexus/thesis_gapfill_patch`
- `codes/nexus/thesis_matrix_v3_ext`
- `codes/nexus/ppo_new_v3_multi_dist`

### 5.2 主结果可用，但整包未完成

判定条件：

- `rl_summary.csv` 与 `rl_trace.csv` 存在
- 无 `DONE.json`

本类主要来自：

- `codes/nexus/local_adaptive_o10_90_s42_rerun2`

### 5.3 开发中 / smoke / 不可作为正式结论

判定条件：

- 仅用于联调
- 预算缩短
- 或缺少最终 `phase4` 实现评估

本类主要来自：

- `codes/nexus/saber_*`

---

## 6. 已测试算法

根据当前代码和已保存结果，已测试的主要算法包括：

- PPO
- A2C
- PPO_NEW v2
- PPO_NEW v3
- PPO_NEW v3.1
- CADM
- NOVA_EDRL
- RARL
- PLR_UED
- CQL_DQN
- SABER_V0 / SABER_V1 / SABER_V1.2 / exploitation 原型

---

## 7. 已完成基线结果（多 seed、完整 run）

以下结果来自 `codes/nexus/thesis_gapfill_patch` 中一组完成的 seed 结果，按分布与算法版本汇总。

### 7.1 O_10_90

| 算法 | 版本 | seeds | average_reward |
|---|---:|---:|---:|
| PPO | v1 | 3 | 0.5100 |
| A2C | v1 | 3 | 0.5033 |
| PPO_NEW | v2 | 3 | 0.5033 |
| PPO_NEW | v3 | 3 | 0.5083 |
| PPO_NEW | v3.1 | 3 | 0.5050 |

对应单 seed 原始值：

- PPO：`[0.505, 0.530, 0.495]`
- PPO_NEW v3：`[0.505, 0.550, 0.470]`
- PPO_NEW v3.1：`[0.505, 0.540, 0.470]`

### 7.2 F1_10_90

| 算法 | 版本 | seeds | average_reward |
|---|---:|---:|---:|
| PPO | v1 | 3 | 0.9133 |
| A2C | v1 | 3 | 0.7950 |
| PPO_NEW | v2 | 3 | 0.9233 |
| PPO_NEW | v3 | 3 | 0.9717 |
| PPO_NEW | v3.1 | 3 | 0.9500 |

### 7.3 G_10_90_60

| 算法 | 版本 | seeds | average_reward |
|---|---:|---:|---:|
| PPO | v1 | 3 | 0.7683 |
| A2C | v1 | 3 | 0.7133 |
| PPO_NEW | v2 | 3 | 0.7917 |
| PPO_NEW | v3 | 3 | 0.8783 |
| PPO_NEW | v3.1 | 3 | 0.8033 |

---

## 8. O_10_90 专项 outer 方法结果（主结果可用，但整包未完成）

以下结果来自：

- `codes/nexus/local_adaptive_o10_90_s42_rerun2`

这些 run 多数无 `DONE.json`，但 `rl_summary.csv` 与 `rl_trace.csv` 存在，可视为主结果可用。

| 算法目录 | inner 算法 | average_reward | 状态说明 |
|---|---|---:|---|
| `run_..._PPO_S42` | PPO v1 | 0.505 | 主结果可用 |
| `run_..._CADM_S42` | PPO_NEW v3 + CADM | 0.495 | 主结果可用 |
| `run_..._NOVA_EDRL_S42` | PPO_NEW v3 | 0.500 | 主结果可用 |
| `run_..._RARL_S42` | PPO_NEW v3 | 0.500 | 主结果可用，但状态文件停留在 running |
| `run_..._PLR_UED_S42` | PPO_NEW v3 | 0.540 | 主结果可用 |
| `run_..._CQL_DQN_S42` | CQL_DQN | 无最终 summary | 不可作为最终结果 |

从这批结果中，`PLR_UED` 是当前 `O_10_90` 上已保存结果里最高的一个主结果。

---

## 9. trace-level 对比（O_10_90）

当前项目已经对 `PPO / PPO_NEW v3 / PPO_NEW v3.1 / PLR_UED` 做过 trace-level 切片分析。  
切片的重点是 implement 阶段的：

- stage / phase
- severity
- semantic action
- hardest removal slice

使用协议化导出指标后，`O_10_90` 的关键 implement 指标可简化为：

| 算法 | avg_reward_implement | Q_hard_rem | R_hard_rem | hard_rem_action1_rate |
|---|---:|---:|---:|---:|
| PPO | 0.5088 | 0.5000 | 0.00746 | 0.00746 |
| PLR_UED | 0.5354 | 0.4667 | 0.05926 | 0.06667 |
| SABER_V0 eval | 0.5051 | 0.3636 | 0.00826 | 0.00826 |
| SABER_V1 eval | 0.5051 | 0.3636 | 0.00826 | 0.00826 |
| SABER_V1.2 eval | 0.5051 | 0.3636 | 0.00826 | 0.00826 |

说明：

- 该表中的 `SABER` 结果来自 smoke/checkpoint 手动评估，不是正式 benchmark。
- `PLR_UED` 是当前唯一在该协议指标上显著高于普通 PPO 的 outer 方法。

---

## 10. 普通 PPO 的适应性实例

普通 PPO 并非在所有分布上都完全坍缩。

项目中的这条历史运行：

- `codes/logs/run_20260213_160505_791293_R30_S3_1_PPO_S42`

其 `rl_summary.csv` 为：

- `average_reward = 0.565`
- `removal_action = 6`
- `removal_wait_action = 189`
- `insertion_action = 4`
- `insertion_non_action = 1`

这说明普通 PPO 在 `S3_1` 上仍会使用两类动作。  
因此，当前项目中的动作坍缩并不是“PPO 永远不会另一动作”，而是分布相关的现象。

---

## 11. 各算法已观测到的客观问题

本节不提供推荐，只记录当前代码和结果中已经出现的事实。

### 11.1 PPO

已观测到的事实：

- 在 `O_10_90` 上，多 seed 平均约 `0.51`
- 在 `S3_1` 上可达到 `0.565`
- 在 `O_10_90` 的 implement trace 中，removal 侧 `action1` 比例非常低

可客观描述的问题：

- hardest OOD 上 removal 侧存在明显保守倾向
- 该现象是分布相关的，不是所有分布都同样严重

### 11.2 PPO_NEW v3 / v3.1

已观测到的事实：

- 在 `F1_10_90` 与 `G_10_90_60` 上，多 seed 平均高于普通 PPO
- 在 `O_10_90` 上，多 seed 平均与 PPO 接近，没有稳定拉开差距

可客观描述的问题：

- 收益具有分布依赖性
- 对 hardest OOD 没有表现出稳定优势

### 11.3 CADM

已观测到的事实：

- `O_10_90` rerun2 的单 seed 结果为 `0.495`
- 低于同目录的 `PPO = 0.505`

可客观描述的问题：

- 在该专项 rerun 中未体现出优于 PPO 的结果

### 11.4 NOVA_EDRL / EDRL

已观测到的事实：

- `O_10_90` rerun2 的单 seed 主结果为 `0.500`
- 与 PPO 接近，没有超过 PPO
- 在 `outer_train_round.csv` 中：
  - 总轮数 `62`
  - `phase2_g_minority` 非零轮数 `0`
  - 高 objective 轮次普遍 `action1_rate = 0`

可客观描述的问题：

- 当前 examined run 中，minority 奖励项未实际发挥作用
- objective 实际上主要由 challenge 项驱动
- outer 选择到的高分 candidate 不等于 minority action 恢复

### 11.5 RARL

已观测到的事实：

- `O_10_90` rerun2 的单 seed 主结果为 `0.500`
- `post_stage/outer_rarl_stats.csv` 最后一行：
  - `dqn_updates = 0`
  - `replay_size = 50`
- `run_status.json` 仍显示：
  - `stage = master`
  - `status = running`

可客观描述的问题：

- examined run 中，outer DQN 没有发生有效更新
- 该 run 的状态文件未显示正常完成

### 11.6 PLR_UED

已观测到的事实：

- `O_10_90` rerun2 的单 seed 主结果为 `0.540`
- `outer_plr_stats.csv` 最后一行显示：
  - `replay_ratio ≈ 0.606`
  - `recent_replay_ratio_w20 ≈ 0.600`
  - `topk_sample_share ≈ 0.600`

可客观描述的问题或限制：

- 当前是专项 rerun 的单 seed 结果
- 整包未完成，没有 `DONE.json`
- 但主结果文件存在，因此可作为当前 best observed OOD result

### 11.7 CQL_DQN

已观测到的事实：

- 目录存在：`run_..._CQL_DQN_S42`
- `rl_trace.csv` 存在
- `rl_summary.csv` 不存在
- `DONE.json` 不存在

可客观描述的问题：

- 当前没有最终 summary，不能作为正式结果纳入比较

### 11.8 SABER 原型

已观测到的事实：

- `SABER_V0 / V1 / V1.2` 的 implement eval 都约为 `0.505`
- 当前多轮 smoke 已记录在：
  - `codes/nexus/saber_v0_smoke`
  - `codes/nexus/saber_v1_smoke`
  - `codes/nexus/saber_v12_smoke`
  - `codes/nexus/saber_v13_exploit_smoke`
- `V13 exploitation` 已触发 sticky replay 与预算放大，但 smoke 中仍只有一次 `R_hard > 0`，且后续 sticky replay 没有继续产生新的 `R_hard > 0`

可客观描述的问题：

- 当前属于开发中原型，不能与已完成 benchmark 直接同等比较
- 目前已完成的 implement eval 未高于 PPO 基线

---

## 12. 当前项目状态概括

基于截至 2026-03-07 的真实代码和结果，可以客观概括为：

1. 项目已经形成完整的：
   - inner RL 训练框架
   - outer 分布选择框架
   - trace / summary / outer round 级日志体系

2. 完整、可信的多 seed 基线已经覆盖：
   - PPO
   - A2C
   - PPO_NEW v2
   - PPO_NEW v3
   - PPO_NEW v3.1

3. 专项 outer 方法在 `O_10_90` 上已有可用结果，但可信度不完全一致：
   - `PLR_UED` 当前单 seed 结果最好
   - `NOVA_EDRL` 与 `RARL` 未显示超过 PPO
   - `CQL_DQN` 本次 rerun 结果不完整

4. 自研 SABER 系列当前仍处于开发联调阶段，已有 smoke 与手动 checkpoint 评估，但尚未形成正式 benchmark 结果。

---

## 13. 结果目录索引（供外部专家直接查验）

### 13.1 完整多 seed 基线

- `codes/nexus/thesis_gapfill_patch`
- `codes/nexus/thesis_matrix_v3_ext`
- `codes/nexus/ppo_new_v3_multi_dist`

### 13.2 O_10_90 专项 outer 方法 rerun

- `codes/nexus/local_adaptive_o10_90_s42_rerun2`

### 13.3 普通 PPO 的适应性示例

- `codes/logs/run_20260213_160505_791293_R30_S3_1_PPO_S42`

### 13.4 SABER 开发性结果

- `codes/nexus/saber_v0_smoke`
- `codes/nexus/saber_v1_smoke`
- `codes/nexus/saber_v12_smoke`
- `codes/nexus/saber_v13_exploit_smoke`

---

## 14. 附：本报告使用的主要代码文件

- `codes/Dynamic_master34959.py`
- `codes/core/dynamic_RL34959.py`
- `codes/experiments/run_experiments_common.py`
- `codes/outer_rl/run_edrl_phase2.py`
- `codes/outer_rl/run_edrl_pipeline.py`
- `codes/algorithms/ppo_new/nn/context_extractor.py`
- `codes/robust_rl/ppo_new/v6_cadm.py`
- `codes/robust_rl/cql_dqn.py`

