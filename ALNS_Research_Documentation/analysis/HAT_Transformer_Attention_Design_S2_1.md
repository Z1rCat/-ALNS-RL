# HAT (Transformer + Attention) 在 S2_1(ABA) 场景下的可实现逻辑设计

## 1. 目标与约束

我们希望在 `S2_1: A -> B -> A` 中，模型在测试阶段重新遇到 A 时，能比普通 PPO 更快恢复到 A 对应策略。

项目约束：

- 动作二分类，单步决策（每次事件触发一次决策）。
- `implement` 阶段参数冻结：不允许梯度更新、不允许 optimizer step。
- 可用信息：`obs`（delay_tolerance, severity, optional event_type）、`stage`（removal/insertion）。
- 可用历史：允许跨 step 保存序列（不在每次 done 后清空）。

这意味着：**适应必须来自前向计算中的“状态依赖机制”，而不是在线学习权重。**

---

## 2. 先澄清：为什么“加了 HAT 不一定必然优于 PPO”

若训练目标和推理规则不一致，HAT 可能被后处理覆盖，表现甚至差于 PPO。常见错误是：

1. 训练时优化的是 `pi(a|h_t)`，推理时改成固定阈值规则（强行改决策边界）。
2. 历史序列虽存在，但推理端实际未正确写入 `prev_action/prev_reward/stage`。
3. 阶段语义冲突（removal 的 action=1 与 insertion 的 action=1 语义不同）没有被显式编码。

因此，目标不是“有 Transformer 就赢”，而是要保证：

- 输入语义正确；
- 训练-推理一致；
- 冻结参数时仍能根据历史改变输出。

---

## 3. 数学定义：可冻结推理的 HAT

### 3.1 Token 定义

第 \(t\) 步 token：

\[
x_t = \left[o_t;\ s_t;\ e(a_{t-1});\ r_{t-1}\right]
\]

其中：

- \(o_t\): 当前观测（连续特征）
- \(s_t \in \{[1,0],[0,1]\}\): stage one-hot
- \(e(a_{t-1}) \in \{[1,0],[0,1]\}\): 上一步动作 one-hot（仅表示编号）
- \(r_{t-1}\in\{0,1\}\): 上一步 reward

长度为 \(K\) 的历史序列：

\[
X_t = [x_{t-K+1}, \dots, x_t]
\]

### 3.2 Transformer 编码

\[
H_t = \mathrm{Transformer}(X_t), \quad H_t \in \mathbb{R}^{K \times d}
\]

使用最后一步 query 做时间注意力聚合：

\[
q_t = W_q h_t^{(K)},\quad
k_i = W_k h_t^{(i)},\quad
v_i = W_v h_t^{(i)}
\]
\[
\alpha_{t,i} = \mathrm{softmax}_i\left(\frac{q_t^\top k_i}{\sqrt d}\right),\quad
c_t = \sum_{i=1}^{K}\alpha_{t,i} v_i
\]

\(c_t\) 是当前决策上下文，已包含历史模式信息。

---

## 4. 关键增强：短期记忆 + 长期原型记忆（冻结可自适应）

单靠短期窗口 \(K\) 在 `ABA` 中容易被中间 B 段污染。需要加入可冻结调用的长期记忆。

### 4.1 长期原型库（训练后固定）

训练阶段构建 \(M\) 个原型向量 \(\{p_j\}_{j=1}^{M}\)（可通过聚类或可学习参数得到）：

\[
w_{t,j} = \mathrm{softmax}_j\left(\frac{\cos(c_t, p_j)}{\tau_m}\right),\quad
m_t = \sum_{j=1}^{M} w_{t,j} p_j
\]

最终上下文：

\[
z_t = [c_t; m_t; s_t]
\]

这里没有在线更新参数；但 \(w_{t,j}\) 随输入变化，推理期仍可切换“更像 A 还是更像 B”。

### 4.2 策略与价值头（阶段条件化）

\[
\pi(a|t) = \pi_{\text{head}(s_t)}(z_t),\quad
V_t = V_{\text{head}(s_t)}(z_t)
\]

其中 `head(s_t)` 表示 removal/insertion 使用不同 head，消除动作语义冲突。

---

## 5. 奖励驱动的注意力调节：正确做法

你的原始想法是“根据 reward 调节 attention”。在冻结约束下，建议做成**训练期监督 + 推理期前向调制**，而不是在线改权重。

### 5.1 训练期：学习“困难度预测”头

定义失败标签 \(f_t = 1-r_t\)。预测头：

\[
\hat f_t = \sigma(w_f^\top z_t)
\]

损失：

\[
\mathcal{L}_{\text{diff}} = \mathrm{BCE}(\hat f_t, f_t)
\]

总损失（示意）：

\[
\mathcal{L} = \mathcal{L}_{\text{PPO}} + \lambda_d \mathcal{L}_{\text{diff}} + \lambda_m \mathcal{L}_{\text{memory}}
\]

### 5.2 推理期：只做前向温度调制，不改参数

\[
T_t = \mathrm{clip}\left(T_0 + \beta \hat f_t,\ T_{\min}, T_{\max}\right)
\]
\[
\tilde{\ell}_t = \frac{\ell_t}{T_t},\quad
\pi_t = \mathrm{softmax}(\tilde{\ell}_t)
\]

这表示“越困难越保守/越平滑”，且不破坏冻结要求。

---

## 6. S2_1(ABA) 为什么这个设计有效

在 `A -> B -> A`：

1. 前段 A：原型注意力 \(w_{t,j}\) 偏向 A 原型。
2. 中段 B：\(w_{t,j}\) 平滑迁移到 B 原型。
3. 后段回到 A：即使短期窗口还残留 B，长期原型匹配会把权重拉回 A 原型，从而恢复 A 对应决策。

简言之，`短期序列 attention` 负责局部动态，`长期原型 attention` 负责模式回忆。

---

## 7. 可直接落地到你工程的实现逻辑（不改主框架）

### 7.1 环境与输入

- 保持现有 `HistoryAttentionWrapper`：
  - token = obs + stage + prev_action + prev_reward
  - `keep_history=1`
- 强制 stage one-hot 注入（已有框架可复用）。

### 7.2 Policy 结构（建议）

1. `AttentionExtractor`: 输出 \(c_t\)。
2. 新增 `PrototypeMemory` 模块：输入 \(c_t\)，输出 \(m_t\) 与 \(w_t\)。
3. 新增 `StageSplitHead`：
  - `actor_removal`, `actor_insertion`
  - `critic_removal`, `critic_insertion`
4. 新增 `difficulty_head` 输出 \(\hat f_t\)。

### 7.3 训练与推理一致性

- 训练时 policy 输出 logits 直接用于 PPO。
- 推理时仅允许：
  - 原型注意力动态变化；
  - 温度调制 \(T_t\)；
  - 不允许额外“硬阈值覆盖规则”替代 policy。

---

## 8. 最小可验证指标（你分析时重点看）

1. 记忆是否在切换：
   - `proto_w_A_mean`, `proto_w_B_mean`（若原型数 >2，记录 top-2）
2. 后段 A 是否恢复：
   - 在 step 350+ 的 `action=1 ratio` 是否向训练前段 A 的比例回归
3. 价值噪声：
   - `value_pred_std`、`advantage_std` 是否在切换后先升后降
4. 推理调制是否工作：
   - `difficulty_hat_mean` 与 `temperature_mean` 的时间曲线

---

## 9. 你现在代码与该设计的差距（直说）

当前实现已经有：

- 历史 token + Transformer；
- stage one-hot；
- implement 历史更新。

当前还缺：

- 长期原型记忆（用于 ABA 回忆）；
- 明确的阶段双 head（当前主线还是共享 head）；
- 训练-推理一致的“软调制”机制（现在有一部分是阈值硬门控）。

所以“最近几次差”不代表 HAT 思路错，而是当前工程形态和你的目标形态还不一致。

---

## 10. 建议的下一步（实现优先级）

1. 先做 `纯HAT基线`（关闭阈值硬门控）验证上限。
2. 加 `PrototypeMemory`（冻结可自适应的核心）。
3. 加 `StageSplitHead`（消除动作语义冲突）。
4. 最后再加 `difficulty temperature`（软调制，不替代 policy）。

这四步每一步都可以单独做 ablation，保证你能回答“到底哪一步贡献了收益”。

