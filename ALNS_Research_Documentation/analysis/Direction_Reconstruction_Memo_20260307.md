# 方向重构备忘录

日期：2026-03-07  
主题：为什么 EDRL 的“以难促另一动作学习”理论在本项目中没有兑现，以及后续主线应如何重构

## 1. 这份备忘录要回答什么

本备忘录只回答四个问题：

1. 我们当初设计 EDRL 的理论直觉是什么。
2. 这套直觉在当前项目里为什么没有兑现。
3. 为什么普通 PPO 有时能表现出适应性，但在 hardest OOD 上仍然会坍缩。
4. 接下来研究主线应如何重述，哪些方向该停，哪些该保留。

结论先写在前面：

> 在本项目中，`低回报` 不是 `另一动作没有学会` 的可靠代理变量。  
> EDRL 当前更多是在选择 `frozen PPO 容易失败的难分布`，而不是选择 `能稳定暴露并修复 minority-action 边界的可学习分布`。  
> 因此，后续主线不应再表述成“更难的 outer 分布让 inner RL 学会另一动作”，而应改写为“更精准地选择 hard-but-learnable 的关键边界 slice，并让后续 exploitation 真正发生”。

---

## 2. 原始 EDRL 理论直觉

当初 EDRL 的核心直觉是成立过一半的：

- 我们的环境中，动作本质上是二元选择。
- 奖励也高度二元化，很多时候可以近似看成“做对/做错”。
- 因此当某类训练分布上的回报低时，一个自然猜想是：
  `当前策略在该分布上没有学会另一种动作，或者该动作使用严重不足。`
- 若 outer RL 以 `hardness` 为目标，就可能把 inner learner 推向“不得不使用另一动作”的分布，进而缓解 action collapse。

这个想法不是空想。它成立的关键前提是：

> `低回报` 必须足够稳定地对应 `关键动作没有学会`。

而当前项目里，真正失效的就是这条前提。

---

## 3. 代码层面的真实目标到底是什么

当前 EDRL 的真实目标函数在：

- `codes/outer_rl/run_edrl_phase2.py`

相关核心逻辑是：

- `phase2_challenge = 1 - J_frozen`
- `phase2_g_minority = max(0, minority_target_rate - phase1_minority_rate)`
- phase2/phase3 objective 近似是：
  `challenge + minority_gain (+ LP) - feasibility_penalty`

这意味着当前 outer 真正在优化的是两类东西：

1. 当前 frozen inner policy 在某个 candidate level 上表现有多差。
2. candidate level 是否让 minority action rate 相比 phase1 baseline 变大。

注意这里的关键偏差：

- `challenge` 只表达“差”，不表达“为什么差”。
- `minority_gain` 用的是全局动作频率，不是关键 slice 上的成功恢复。

所以代码实现层面，EDRL 从一开始就不是在优化：

`高 severity 关键边界上另一动作的正确恢复`

而是在优化：

`整体更难 + 动作比例相对更平衡`

这两者不是一回事。

---

## 4. 为什么这套理论在真实运行里没兑现

### 4.1 低回报不是“另一动作没学会”的单一信号

在当前项目里，低回报至少可能来自以下几种不同机制：

- 当前状态本来就该保守，但 PPO 连保守都做错了。
- 当前状态该出手，但 PPO 没出手。
- 该 candidate level 本身太难，两个动作都容易错。
- candidate level 里真正需要 minority action 的状态占比太低，整体回报低，但没有给 inner learner 足够多的关键正样本。
- removal 和 insertion 两个 stream 里的 `0/1` 语义不对称，混在一起统计会进一步稀释“另一动作”的真实含义。

因此：

> `低回报` 只能说明“这个 level 对当前策略困难”，不能说明“它能教会另一动作”。

这就是 EDRL 理论失败的第一层原因。

### 4.2 真实运行里，minority 项几乎没有发挥作用

对这次 `O_10_90` 的 `NOVA_EDRL`：

- 路径：`codes/nexus/local_adaptive_o10_90_s42_rerun2/run_20260305_092522_719454_R30_O_10_90_NOVA_EDRL_S42/post_stage/outer_train_round.csv`

汇总后得到：

- 总轮数：`62`
- `objective_score` 非零轮数：`62`
- `action1_rate` 非零轮数：`16`
- `phase2_g_minority` 非零轮数：`0`

这说明本次 EDRL 运行里，outer 几乎一直只在优化 `challenge`，并没有真正吃到“minority 恢复”的奖励。

更直观地看，高 objective 的那些 candidate，很多都满足：

- `action1_rate = 0`
- `minority_rate = 0`
- `phase2_g_minority = 0`

也就是说，outer 选中的往往只是“当前 PPO 很难”的 level，而不是“能推动 minority action 学习”的 level。

这就是 EDRL 理论失败的第二层原因。

### 4.3 EDRL 使用的是全局 minority rate，而不是关键 slice 成功率

当前 outer 使用的 `minority_rate` 本质上来自训练决策中：

- `action0_rate`
- `action1_rate`
- 二者较小者

这在 `run_edrl_phase2.py` 的 `_calc_metrics(...)` 里可以看到。

问题是，真正决定 OOD 性能的，不是“全局 action1 多不多”，而是：

`高 severity × removal` 这类关键 slice 上，少量该出手的样本有没有被正确恢复。

全局 minority rate 会把大量无关样本也算进去，导致两个后果：

- outer 可能偏好“整体动作更平衡，但关键边界仍然错”的 candidate；
- outer 也可能惩罚那些“全局动作仍然很偏，但关键边界修复了”的 candidate。

因此，EDRL 的统计对象和真正的 failure mode 并不对齐。

这就是第三层原因。

### 4.4 难度和可学性被混在一起了

EDRL 假定：

`更难 -> 更能逼 inner learner 学会另一动作`

但当前项目里，hardness 和 learnability 不是同一个量。

很多 candidate level 满足：

- `J_frozen` 很低
- 但并不提供足够清晰、足够密集的 minority-action 正样本
- inner learner 在其上最稳的解仍然是继续坍缩到 `wait`

于是形成一个错误闭环：

1. outer 挑到 `很难` 的 level；
2. inner 在这些 level 上仍然学到保守坍缩解；
3. outer 继续认为这些 level “很有挑战性”，于是继续重放；
4. 最终 outer 很忙，但 inner 还是 PPO 的坍缩解。

这就是第四层原因。

### 4.5 removal / insertion 语义不对称，但 EDRL 没有显式区分

在当前项目里：

- removal stream 的 `action=0` 更接近 `wait`
- insertion stream 的 `action=0` 更接近 `insert`

也就是说，同一个 `0/1` 在两个 stream 里不是同义动作。

但 EDRL 的 minority 统计基本是在更粗粒度上算的，没有把：

- 哪个 stream
- 哪种语义动作
- 哪个 severity slice

拆开。

所以 EDRL 以为自己在“鼓励另一动作”，但实际上混合了不同 stream 的不同语义。

这就是第五层原因。

---

## 5. 为什么普通 PPO 有时也能表现出适应性

这点很关键，因为它说明问题不是“PPO 永远学不会另一动作”。

普通 PPO 的这条运行：

- `codes/logs/run_20260213_160505_791293_R30_S3_1_PPO_S42`

它的 `rl_summary.csv` 显示：

- `average_reward = 0.565`
- `removal_action = 6`
- `insertion_action = 4`
- `insertion_non_action = 1`

这说明普通 PPO 在 `S3_1` 这种分布上，确实会使用两类动作，不是绝对坍缩。

但对 `O_10_90` 的 PPO：

- `codes/nexus/thesis_gapfill_patch/run_20260219_202213_305730_R30_O_10_90_PPO_S42/rl_summary.csv`

则表现为：

- `average_reward ≈ 0.505`
- removal 侧几乎完全坍缩到 `wait`

这说明：

> PPO 不是“不会另一动作”，而是“在 hardest OOD 的关键边界上，不愿意稳定跨过那条边界”。

换句话说，PPO 的问题不是动作能力缺失，而是：

- 在较容易或较干净的分布里，它本来就会 action1；
- 在 hardest high-severity boundary 上，它缺乏足够稳定、足够密集、足够可学的训练经历，于是退回到风险更低的保守解。

这是方向判断里最重要的分界。

---

## 6. trace-level 证据进一步说明了什么

我们已经做过 `O_10_90` 的 trace-level 切片对比。结论很稳定：

- `PLR_UED` 的收益不是“全局都略微提升”；
- 它主要修复的是：
  `high severity × removal`
- 其次带来少量 insertion 侧收益；
- 它的关键作用是：
  让高 severity 边界上出现少量但高质量的 `action1`，并让保守动作也更准。

这和 EDRL 的 outer 目标有本质区别：

- `PLR_UED` 更像是在做“有用的训练分布调度”；
- `EDRL` 更像是在做“全局困难度追逐”。

所以当前证据支持的不是：

`只要 outer 更强，inner 就会被逼出另一动作`

而是：

`只有 outer 真正把训练分布对准 hardest boundary，inner 才有机会跨过动作边界`

---

## 7. 最近 SABER 试验给我们的额外信息

最近 `SABER_V1 / V1.1 / V1.2 / exploit` 这几轮尝试，还带来一条额外结论：

> 即便 outer 已经比以前更会挑 candidate，若后续 exploitation 不够及时、不够集中，最终 implement 行为仍可能和普通 PPO 完全一样。

实际现象是：

- `SABER_V1.2` 已经能把 top candidate 排到 `R_hard > 0` 的候选上；
- 但手动 phase4 eval 后，最终 implement 行为仍和 `SABER_V0 / PPO` 几乎逐行一致；
- 后续 `sticky replay + budget boost` 虽然触发了，但仍没有把正确信号转化成持续的 `R_hard > 0`。

这说明当前问题至少分成两段：

1. 选对 candidate。
2. 把 candidate 的局部正确信号转成稳定的策略边界移动。

EDRL 主要卡在第 1 段，SABER 近期试验则说明第 2 段同样不能忽略。

---

## 8. 方向重构：哪些判断应当保留

以下判断现在应当保留：

### 8.1 “训练分布选择/调度”仍然是主线

这个判断没有变，反而更强了。

原因：

- `PLR_UED` 是当前唯一稳定优于 PPO 的 outer 系方法；
- `EDRL` 和 `RARL` 没有表现出更强 outer 控制器应有的优势；
- `PPO_NEW v3/v3.1` 单独也不能稳定解决 `O_10_90`。

因此主线仍然应是：

`更聪明的训练分布选择 / replay / curriculum`

而不是：

`更复杂的 outer RL controller`

### 8.2 真正要修的是 hardest boundary，不是全局动作比例

后续目标不该写成：

`提升 minority action rate`

而应写成：

`修复 high severity × removal 等关键边界 slice 上的 selective action1 recovery`

### 8.3 hard 必须和 learnable 一起定义

后续 outer 目标必须同时考虑：

- 这个 level 会不会暴露关键错误；
- 它是不是仍然可学；
- 它是否真的提供了关键 slice 的正样本，而不是只提供总体困难度。

---

## 9. 方向重构：哪些表述应当废弃

以下旧表述现在不建议继续作为主理论：

### 9.1 “低回报 level 会逼 inner learner 学另一动作”

这个说法太强，而且在当前项目里不成立。

更准确的说法应该是：

`只有当低回报主要来自关键动作缺失，并且该 candidate 仍然可学时，它才可能推动另一动作学习。`

### 9.2 “动作坍缩主要可以用全局 minority rate 来刻画”

这个说法不够精确。

当前项目里，动作坍缩是 slice-specific 的，至少要按：

- stage / stream
- severity
- semantic action

拆开看。

### 9.3 “outer 只要更难，就会更有效”

这个说法现在应当明确放弃。

应该改成：

`outer 的价值不在于更难，而在于更准确地暴露关键且可学习的错误模式。`

---

## 10. 新主线应该如何表述

建议将研究主线重述为：

> 在该环境中，OOD 失败主要表现为少量高严重度关键边界上的动作坍缩，而不是全局动作能力缺失。  
> 因此，外层训练分布选择不应以总体困难度为目标，而应以关键边界 slice 的暴露、可学习性和后续 exploitation 效率为目标。

这比原来的 EDRL 表述更准确，也更贴近现有证据。

---

## 11. 对后续算法设计的直接启示

基于当前所有证据，后续算法设计应优先遵守四条原则：

### 11.1 外层目标必须直接对应关键 slice

不要再用：

- 全局低回报
- 全局 minority rate

去间接代理关键边界。

应尽量直接使用：

- `Q_hard`
- `R_hard`
- `P_easy`
- stream-aware / severity-aware 指标

### 11.2 removal 和 insertion 必须非对称处理

它们的动作语义不同，后续 outer score 和 diagnostics 不能再用统一二元动作表述去覆盖。

### 11.3 candidate 选择和 exploitation 要分开建模

当前经验已经说明：

- 只会选 candidate 不够；
- 选到了 candidate 之后，还要让成功信号被及时、持续放大。

也就是说，outer 机制至少要拆成：

1. candidate ranking
2. success-triggered exploitation

### 11.4 评估时必须按 trace-level slice 报告

后续任何新算法，都不该只报 `average_reward`。

至少要同时报：

- hardest removal slice 的 `Q_hard`
- hardest removal slice 的 `R_hard`
- easy slice preservation
- implement 行为是否真的跨过离散动作边界

---

## 12. 建议立即停止和继续的事项

### 12.1 建议停止

- 不再继续把 `challenge = 1 - J_frozen` 当作主要 outer 信号。
- 不再把“低回报”直接解释成“需要更多另一动作”。
- 不再把全局 minority rate 当成主诊断指标。
- 不再优先探索更复杂的 outer RL 控制器。

### 12.2 建议继续

- 继续坚持“训练分布选择/调度”为主线。
- 继续做 trace-level hardest-slice 诊断。
- 继续做 stream-aware、severity-aware、hard-but-learnable 的 outer score。
- 继续研究 success-triggered exploitation，但要重新设计其触发时机和锚定方式。

---

## 13. 当前最合理的研究结论口径

如果要把现在的理解写成论文或设计报告中的正式口径，建议使用下面这版：

> 在本项目中，普通 PPO 并非完全缺乏 minority action 能力；其失败主要集中在 hardest OOD 的高严重度边界 slice 上。  
> 现有 EDRL 以总体困难度为核心 outer 目标，但低回报在该环境中并不能可靠区分“关键动作缺失”与“整体不可学困难”，因此未能稳定缓解动作坍缩。  
> 相比之下，现有证据更支持将后续方法设计为面向关键边界 slice 的训练分布选择与 exploitation 机制，而不是继续增强通用 outer RL 控制器。

---

## 14. 一句话版本

一句话概括本次方向重构：

> 以后不要再问“怎样让 outer 找到更难的 level”，而要问“怎样让 outer 找到能稳定暴露并修复 hardest boundary 的可学习 level，并把这类 level 的局部成功真正转化为策略边界移动”。  

