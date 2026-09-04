# Adp2 实验执行方案：从局部解释片段形成稳定、最小、可复用的 \(Z_D\)

> **文档定位**  
> 本文档用于指导工程师在 E0 与 Adp1 已完成的基础上执行下一阶段实验。  
> Adp2 不直接进入 relation \(R_D\)，也不进入视觉、TP discovery、任务层。  
> 它只回答一个更具体的问题：
>
> \[
> \boxed{
> M_{\max}>M_{\mathrm{real}}
> \text{ 且初始 candidate 没有稳定身份时，}
> }
> \]
>
> \[
> \boxed{
> \text{能否仅依靠 interaction 中反复出现的 transformation 规律，}
> \text{把局部解释片段逐步整理成正确数量、正确身份的 }Z_D？
> }
> \]

---

# 0. 一页执行摘要

现有实验已经得到以下事实：

1. **O1 PASS**：如果真实 \(m,v\) 已知，当前 mechanism network 能几乎完美拟合真实 mechanism。
2. **O2 PASS**：如果给监督，当前 \(q_\eta(S,p)\) 有能力恢复真实 \(m,v\)。
3. **O3-E PASS**：当真实 mechanism basis 已知时，枚举 support 后，当前 effect + local sparsity 目标在 99.7% 样本上偏好真实 participation。
4. **O5 PASS**：如果从接近真实的 mechanism basin 初始化，结构能够稳定，并能压掉多余 candidate。
5. **原始 E0 FAIL**：从随机初始化联合训练时，\(M_{\max}=5\) 个 candidate 会形成共同重构。
6. **Adp1 FAIL**：即使每轮精确枚举 support，并把 assignment 与 mechanism update 分开，candidate identity 仍持续重排；每个样本只用约 2--3 个 candidate，但整个数据集中五个 candidate 都被使用，且 functional recovery 失败。

因此 Adp2 不再把问题简单表述为“如何让 5 个 candidate 中死掉 2 个”，而拆成两个子问题：

\[
\boxed{
A:\ \text{一个 candidate 如何获得稳定的 mechanism identity？}
}
\]

\[
\boxed{
B:\ \text{在 identity 形成后，模型如何在不知道 }M_{\mathrm{real}}\text{ 的情况下保留正确数量的 mechanism？}
}
\]

Adp2 的核心思想是：

\[
\boxed{
\text{少，不够；还必须跨 interaction、跨 context 可复用。}
}
\]

并且：

\[
\boxed{
\text{不确定的 assignment 不应立即塑造 mechanism identity。}
}
\]

最终训练逻辑为：

```text
局部解释
→ 证据筛选
→ 跨 context 复用检验
→ identity 稳定
→ 候选合并 / 必要时拆分
→ 全局机制数量收缩
```

---

# 1. Adp2 不能修改的基础条件

除本文明确列出的 Adp2 新增步骤外，必须沿用原 E0/Adp1 的 world、数据和 mechanism 结构。

## 1.1 Ground-truth world

```text
WORLD_SEED = 20260901
DATASET_SEED = 20260902
M_real = 3          # 只允许 evaluator 读取
M_max = 5           # learner 只知道这个上限
state slots = 3
slot dim = 2
v dim = 1
TP = identity       # Adp2 第一阶段固定 identity
```

真实 mechanism：

\[
\Delta S=e_1+e_2+e_3.
\]

真实 interaction participation 与原 E0 完全一致，训练中继续不出现：

\[
\boxed{101}
\]

作为组合泛化测试。

## 1.2 Learner 可见信息

训练阶段只允许使用：

\[
\boxed{(S_i,p_i,\Delta S_i)}
\]

以及 learner 自己产生的：

\[
m_i,\ v_i,\ \mathcal M_j.
\]

禁止使用：

\[
m_i^*,\quad v_i^*,\quad M_{\mathrm{real}},\quad e_i^{j*}
\]

做训练决策。

这些 GT 只允许用于最终评价。

## 1.3 Mechanism bank

继续使用原来的：

```text
SetMechanismBank
token_dim = 64
attention_heads = 4
cross_attention_layers = 2
M_max = 5
```

不得在 Adp2 中同时更换 mechanism architecture。

## 1.4 暂时不加入的东西

Adp2 第一轮禁止加入：

```text
R_D relation
视觉编码
TP segmentation
Gaussian v
nonlinear TP
额外 observation noise
正交约束
独立性约束
去相关约束
对比学习约束
任务 reward
```

原因：当前只验证 mechanism identity formation 与 model-order selection。

---

# 2. Adp1 已经证明了什么

Adp1 的 seed 0 出现：

```text
IID NRMSE = 0.3633
participation F1 = 0.5714
min instance-effect R² = -0.2813
min functional R² = -0.4020
SCR ≈ 0.20 ~ 0.24
```

最终五个 candidate 的 population usage：

```text
[0.506, 0.529, 0.488, 0.506, 0.473]
```

但每个样本平均只选择约：

\[
2.4
\]

个 candidate。

因此 Adp1 的失败不是“每个样本太稠密”，而是：

\[
\boxed{
\text{sample-level sparse}
\not\Rightarrow
\text{world-level stable mechanisms}
}
\]

更具体地：

1. 当前 basis 决定本轮 assignment；
2. 本轮 assignment 又决定 candidate 下一轮学什么；
3. candidate function 改变后，下一轮 assignment 再变化；
4. 整个系统形成“局部稀疏，但身份持续漂移”的状态。

Adp2 必须直接针对这个失败模式。

---

# 3. Adp2 的两个待验证假设

## 3.1 假设 A：mechanism identity 来自跨 interaction 的复用

一个 candidate 不能因为“在自己训练过的样本上能降低 reconstruction error”就被认为是一个 \(Z_D\)。

只有当它在不同 interaction、不同 state context 中仍能以同一套 mechanism function 解释对应变化时，才应逐渐获得稳定 identity。

因此：

\[
\boxed{
Z_D^j\text{ 的身份依据应是跨 context 的重复解释能力。}
}
\]

## 3.2 假设 B：决定 \(M_{\mathrm{real}}\) 需要全局机制成本

原来的 local sparsity 只惩罚：

\[
\sum_{i,j}m_i^j.
\]

它只能要求：

> 一个 sample 少用几个 candidate。

它不能阻止：

> 整个数据集把 5 个 candidate 全部养活，只是不同 sample 轮流使用。

因此需要额外区分：

\[
g_j=
\begin{cases}
1,&\text{candidate }j\text{ 被当前 world model 保留}\\
0,&\text{candidate }j\text{ 当前不作为 mechanism 存在}
\end{cases}
\]

以及：

\[
m_i^j=\text{已保留 mechanism }j\text{ 在 interaction }i\text{ 是否参与}.
\]

要求：

\[
\boxed{m_i^j\le g_j.}
\]

这里 \(g_j\) 不是世界物理变量，只是 learner 的“候选槽位是否保留”状态。

---

# 4. Adp2 总体实验路线

不要一次把所有新机制全部打开。

必须按以下顺序执行：

```text
Adp2-0  日志与基线复现
   ↓
Adp2-A0 跨 context 复用指标校准（只诊断，不改训练）
   ↓
Adp2-A1 只加入“低证据 assignment 暂缓”
   ↓
Adp2-A2 再加入“assignment 切换门槛”
   ↓
Adp2-A3 再加入“跨 context 复用检验”
   ↓
Adp2-B0 全局机制成本 + 合并逻辑的受控 sanity test
   ↓
Adp2-B1 随机初始化下加入全局收缩与合并
   ↓
Adp2-S  只有出现过度合并时才做拆分诊断
   ↓
Adp2-F  完整多 seed 确认
```

原则：

\[
\boxed{\text{每一步只增加一个新的决策依据。}}
\]

任何一步 FAIL，都先解释 FAIL，不要直接跳到更复杂版本。

---

# 5. Adp2-0：先补齐日志并复现 Adp1

## 5.1 目的

Adp1 已经暴露一个严重问题：很多关键结构指标没有逐轮保存。

Adp2 在改变算法之前必须先确保每轮可以观察：

- assignment 是否稳定；
- 每个 candidate 是否逐渐形成固定样本群；
- candidate 是否跨 context 复用；
- candidate 数量是否真的减少；
- 哪一步发生错误合并或错误拆分。

## 5.2 每轮必须保存

```text
round_metrics.jsonl
```

每一行至少包含：

```text
round
train_effect_mse
train_objective
val_nrmse
test_iid_nrmse        # 可每 2~3 round 评一次
expected_active_per_sample

candidate_usage[5]
candidate_effect_norm[5]

support_change_rate_bit
support_exact_agreement
support_cardinality_histogram[0..5]
top_support_patterns

assignment_margin_mean
assignment_margin_median
assignment_margin_p10
assignment_margin_p90
resolved_fraction

reuse_error_per_candidate[5]
reuse_context_coverage[5]

global_alive[5]
num_alive

merge_proposals
merge_accepted
split_proposals
split_accepted

e_step_seconds
m_step_seconds
reuse_check_seconds
merge_check_seconds
```

同时每轮保存：

```text
checkpoints/round_xxx_bank.pt
assignments/round_xxx.npz
```

## 5.3 基线

用 Adp1 原设置跑 seed 0：

```text
train = 1024
val = 256
test_iid = 512
test_context = 512
test_101 = 512
rounds = 12
```

要求关键结果与已有 Adp1 报告误差在合理范围内。

如果复现失败，STOP。

---

# 6. Context 分组：不使用 GT 的“不同情境”

Adp2 需要检查 candidate 是否能跨不同 state context 复用。

不能使用真实 mechanism label 来定义 context。

## 6.1 第一版分组方法

将：

\[
S\in\mathbb R^{3\times2}
\]

展平：

\[
x=\operatorname{vec}(S)\in\mathbb R^6.
\]

固定：

```text
CONTEXT_SEED = 20260903
```

生成两个固定随机方向：

\[
q_1,q_2\in\mathbb R^6,
\qquad
q_1^\top q_2=0,
\qquad
\|q_1\|=\|q_2\|=1.
\]

计算：

\[
c_1=q_1^\top x,
\qquad
c_2=q_2^\top x.
\]

只使用 train split 计算：

\[
\operatorname{median}(c_1),\quad
\operatorname{median}(c_2).
\]

由正负两侧形成四个 context group：

```text
G00
G01
G10
G11
```

之后 val/test 使用同一 threshold，不重新计算。

## 6.2 为什么这样分

它只依赖 observable state \(S\)，不使用 GT \(m,v\)。

目的不是声称真实世界天然有四种 context，而是人为构造：

\[
\boxed{\text{互不重叠的 state 区域}}
\]

来检查同一 mechanism 是否能跨状态区域工作。

---

# 7. Adp2-A0：先验证“跨 context 复用指标”有没有辨别力

> **这是 Adp2 最重要的前置诊断。**
>
> 如果我们连“正确 mechanism 比错误 fragment 更可复用”都测不出来，
> 就不能把这个量放进训练过程。

## 7.1 正样本与负样本

使用已有两个 checkpoint：

### 正样本

O5-B 的 GT-like basin：

```text
M_max = 5
前三个 candidate 已恢复 GT-like mechanisms
candidate 4/5 被压到接近 0
```

只评估前三个已匹配 candidate。

### 负样本

Adp1 seed 0 failed checkpoint：

```text
五个 candidate 均参与
functional recovery 为负
identity 漂移
```

## 7.2 复用误差定义

对于 candidate \(j\)，选择一个 context group \(G_g\) 作为 held-out group。

其它三个 group 用于当前 candidate 的 mechanism fitting：

\[
\mathcal D_j^{train}
=
\{i:m_i^j=1,\ context(i)\neq g\}.
\]

在 held-out group：

\[
\mathcal D_j^{test}
=
\{i:m_i^j=1,\ context(i)=g\}.
\]

对 test sample \(i\)，其它 candidate 固定，构造 residual：

\[
r_{i,-j}
=
\Delta S_i
-
\sum_{k\neq j}
m_i^k\mathcal M_k(S_i,v_i^k).
\]

**不允许更新 \(\mathcal M_j\) 参数。**

只允许重新求该 sample 的 realization：

\[
\tilde v_i^j
=
\arg\min_v
\left\|
r_{i,-j}
-
\mathcal M_j(S_i,v)
\right\|^2.
\]

得到 held-out reuse error：

\[
\boxed{
E_{j,g}^{reuse}
=
\frac{
\sum_{i\in\mathcal D_j^{test}}
\|
r_{i,-j}
-
\mathcal M_j(S_i,\tilde v_i^j)
\|^2
}{
\sum_{i\in\mathcal D_j^{test}}
\|r_{i,-j}\|^2+\epsilon
}
}
\]

最终：

\[
\boxed{
E_j^{reuse}
=
\operatorname{median}_{g\in\{00,01,10,11\}}
E_{j,g}^{reuse}
}
\]

同时记录 candidate 覆盖了几个 context group。

## 7.3 关键要求

candidate 至少要在：

\[
\boxed{3/4}
\]

个 context group 中有足够样本，才允许讨论“可复用”。

第一版：

```text
n_min_per_group = 32
```

不足则标记：

```text
insufficient evidence
```

而不是直接判定为差 mechanism。

## 7.4 A0 PASS 条件

不预先硬写 reuse threshold。

先观察：

```text
O5 GT-like candidate reuse-error distribution
vs
Adp1 failed candidate reuse-error distribution
```

要求至少满足：

\[
Q_{90}(E_{\mathrm{O5}})
<
Q_{10}(E_{\mathrm{Adp1}})
\]

即正、负两组存在清晰间隔。

若存在间隔，设置：

\[
\boxed{
\tau_{reuse}
=
\frac{
Q_{90}(E_{\mathrm{O5}})
+
Q_{10}(E_{\mathrm{Adp1}})
}{2}
}
\]

这个阈值以后固定，不允许根据 Adp2 最终 GT recovery 调整。

### 如果 A0 FAIL

STOP。

说明当前 reuse metric 本身没有辨别力。

下一步应重新设计“可复用”的观测方式，而不是把它强行加入训练。

---

# 8. 每个 assignment 的“证据强度”

Adp1 对每个 sample 都必须选一个 top-1 support。

Adp2 要区分：

\[
\boxed{\text{最优解释}}
\]

与：

\[
\boxed{\text{明显优于其它解释的最优解释}}
\]

## 8.1 E-step 保持与 Adp1 相同

对：

\[
M_{\max}=5
\]

枚举：

\[
2^5=32
\]

个 support。

每个 support 内优化 active \(v\)。

得到：

\[
J_i^{(1)}
\le
J_i^{(2)}
\le\dots
\]

## 8.2 相对 margin

定义：

\[
\boxed{
\Delta_i
=
\frac{
J_i^{(2)}-J_i^{(1)}
}{
|J_i^{(1)}|+\epsilon
}
}
\]

\(\Delta_i\) 大：

> 当前 basis 下第一名解释明显优于第二名。

\(\Delta_i\) 小：

> learner 自己也分不清哪种解释更合理。

## 8.3 resolved / unresolved

第一版 primary：

```text
tau_conf = 0.05
```

若：

\[
\Delta_i\ge\tau_{conf}
\]

标记：

```text
resolved
```

否则：

```text
unresolved
```

unresolved sample：

- 可以参与总 reconstruction evaluation；
- 不能用于 mechanism identity fitting；
- 不能触发 merge；
- 不能触发 candidate death。

## 8.4 小规模敏感性检查

只在 seed 0 development run 做：

```text
tau_conf ∈ {0.02, 0.05, 0.10}
```

目标不是“调出最好 GT”，而是确认：

- threshold 不应只有一个极窄值有效；
- resolved fraction 不应接近 0 或 1；
- 更高 threshold 应提高 pseudo-assignment 的 GT precision（只做 evaluator 统计）。

若只有一个非常窄 threshold 才有效，则当前证据机制不稳健。

---

# 9. Adp2-A1：只加入“低证据 assignment 暂缓”

## 9.1 唯一变化

相对于 Adp1：

```text
E-step：不变
M-step：只使用 resolved samples
其余全部不变
```

不加入：

```text
assignment switch threshold
reuse gating
global mechanism cost
merge
split
```

## 9.2 目的

回答：

\[
\boxed{
\text{Adp1 是否主要因为把低证据 top-1 当成确定事实，导致错误 identity 自我强化？}
}
\]

## 9.3 预期现象

如果想法正确，应看到：

1. SCR 比 Adp1 明显下降；
2. candidate usage 不再五个都约 0.5；
3. functional \(R^2\) 开始从负值向正值移动；
4. resolved fraction 逐步上升，而不是持续下降；
5. prediction 可能早期下降更慢，这是允许的。

## 9.4 A1 阶段性 PASS

Development seed 0 相对 Adp1 至少满足：

```text
末 3 轮 SCR 平均下降 >= 30%
min matched functional R² 提高 >= 0.30
resolved fraction 不低于 0.20
不存在 candidate 全部死亡
```

这不是最终 E0 PASS，只表示“暂缓低证据 assignment”值得继续。

---

# 10. Adp2-A2：加入 assignment 切换门槛

## 10.1 动机

Adp1 每轮都重新：

\[
m_i^{new}
=
\arg\min_mJ_i(m).
\]

即使新 support 只比旧 support 好一点点，也会立即换身份。

Adp2-A2 要求：

> 新解释必须明显更好，才允许推翻上一轮已经使用的解释。

## 10.2 定义

记旧 support：

\[
m_i^{old}.
\]

重新计算其当前 objective：

\[
J_i^{old}.
\]

新 top-1：

\[
m_i^{new},
\quad
J_i^{new}.
\]

定义改善比例：

\[
\boxed{
I_i
=
\frac{
J_i^{old}-J_i^{new}
}{
|J_i^{old}|+\epsilon
}
}
\]

只有同时满足：

\[
\Delta_i\ge\tau_{conf}
\]

和：

\[
I_i\ge\tau_{switch}
\]

才允许：

\[
m_i^{old}\rightarrow m_i^{new}.
\]

第一版：

```text
tau_switch = 0.02
```

如果 sample 之前没有 resolved assignment，则只看 \(\tau_{conf}\)。

## 10.3 目的

只验证：

\[
\boxed{
\text{减少由极小 objective 波动引起的 identity 抖动，是否能促进稳定 specialization？}
}
\]

## 10.4 关键指标

额外记录：

```text
proposed_switch_fraction
accepted_switch_fraction
rejected_small_gain_switch_fraction
```

以及：

\[
A^{(r)}
=
\frac1N
\sum_i
\mathbf1[m_i^{(r)}=m_i^{(r-1)}].
\]

## 10.5 A2 阶段性 PASS

相对 A1：

```text
末 3 轮 SCR 再下降 >= 20%
exact support agreement 上升
functional R² 不下降
IID NRMSE 不恶化超过 10%
```

如果只让 assignment 冻住，但 functional \(R^2\) 仍很差，则说明只是“稳定了错误分法”。

这种情况不能算 PASS。

---

# 11. Adp2-A3：真正加入“跨 context 可复用性”

A3 是问题 A 的核心实验。

## 11.1 candidate evidence pool

对每个 candidate \(j\)，只保存：

```text
resolved
+
stable for at least 2 rounds
+
m_i^j = 1
```

的 sample。

记为：

\[
\mathcal E_j.
\]

这些 sample 是 candidate 当前的“身份依据”。

## 11.2 稳定身份条件

candidate \(j\) 只有同时满足：

1. evidence pool 样本数足够；
2. 覆盖至少 3 个 context group；
3. reuse error：

\[
E_j^{reuse}<\tau_{reuse}
\]

连续：

```text
P_stable = 3 rounds
```

才标记为：

```text
stable_candidate
```

否则：

```text
provisional_candidate
```

## 11.3 stable candidate 与 provisional candidate 的区别

### stable candidate

- 可以作为其它 sample 的稳定解释；
- mechanism update 只允许小步更新；
- 不因为一轮 usage 低就删除；
- 可以参与 merge test。

### provisional candidate

- 可以继续学习；
- identity 还未被确认；
- 不能触发其它 candidate death；
- 可以被重新分配、被合并。

## 11.4 保守更新

A3 不增加新的 loss。

只限制每轮 mechanism 参数变化速度。

Primary：

```text
optimizer = AdamW
lr = 1e-4
steps_per_candidate_per_round = 8
gradient_clip = 1.0
```

如果 candidate 已 stable：

```text
lr = 5e-5
steps = 4
```

每轮计算 function drift：

\[
D_j^{drift}
=
E_{S,v}
\left[
\|
\mathcal M_j^{(r+1)}(S,v)
-
\mathcal M_j^{(r)}(S,v)
\|^2
\right]
\]

只把它作为日志和 early-stop 依据，不加入 loss。

## 11.5 A3 的关键问题

\[
\boxed{
\text{一个 candidate 一旦在多个 context 中被证明可复用，}
\text{是否会开始形成稳定的 world-level identity？}
}
\]

## 11.6 A3 阶段性 PASS

Development seed 0：

```text
至少 3 个 candidate 成为 stable
至少 2 个 candidate 保持 provisional 或低 usage
末 3 轮 SCR < 0.10
min functional R² > 0.50
mean participation F1 > 0.75
```

进入正式确认前，最终仍必须满足原 E0 阈值。

---

# 12. 问题 B：全局 mechanism 是否值得“长期养着”

从 A3 开始，才允许处理 mechanism 数量。

不能在 identity 还没形成时直接按 usage 删 candidate。

---

# 13. 全局存在状态 \(g_j\)

维护：

```text
g_j ∈ {0, 1}
```

其中：

```text
1 = 当前 world model 保留该 candidate
0 = dormant，不参与 prediction 和 assignment
```

要求：

\[
m_i^j\le g_j.
\]

初始：

\[
g_j=1,\qquad j=1,\dots,5.
\]

## 13.1 全局 objective

使用平均 reconstruction，使规模不随样本数变化：

\[
\boxed{
J_{\mathrm{global}}
=
L_{\mathrm{effect}}
+
\lambda_{\mathrm{local}}
\frac1N\sum_i\|m_i\|_0
+
\lambda_G
\sum_jg_j
}
\]

其中：

```text
lambda_local = 1e-3
```

保持原设置。

\(\lambda_G\) 是“长期保留一个 mechanism 的成本”。

它不是物理定律，也不是世界变量，只是当 learner 不知道 \(M_{\mathrm{real}}\) 时用于比较复杂 world model 与简单 world model 的模型选择成本。

---

# 14. Adp2-B0：先做受控的“重复 mechanism” sanity test

> 不要直接在 random-init full discovery 上验证合并。  
> 先证明合并逻辑至少能处理一个已知的重复 mechanism。

## 14.1 构造

从 O5-B 的成功 checkpoint 出发。

保留三个 GT-like candidate：

```text
C1, C2, C3
```

然后人为复制：

```text
C4 = copy(C1)
```

将原来属于 C1 的训练 assignment 随机分成两半：

```text
一半给 C1
一半给 C4
```

这样 prediction 基本不变，但当前 model 显式拥有：

\[
4
\]

个有效 candidate，其中 C1 与 C4 本质重复。

C5 保持 dormant。

learner 的 merge 逻辑禁止读取“C4 是复制出来的”这一信息。

## 14.2 目的

直接回答：

\[
\boxed{
\text{当两个 candidate 真的是同一 mechanism 的重复版本时，}
\text{全局存在成本 + held-out merge test 能否把它们恢复成一个？}
}
\]

---

# 15. Pairwise merge test

对当前 alive candidate \(j,k\)：

## 15.1 什么时候允许提出 merge

至少满足：

```text
二者都有足够 evidence
二者 evidence pool 总样本数 >= 256
二者 union 覆盖 >= 3 个 context groups
```

## 15.2 临时 merged candidate

构造：

\[
\mathcal M_{j\oplus k}.
\]

初始化使用：

```text
reuse error 更低的那个 candidate 参数
```

只在：

\[
\mathcal E_j\cup\mathcal E_k
\]

的 train 部分训练。

每个 context group 轮流 held-out。

训练时其它 candidate 固定。

## 15.3 merged model 的 validation objective

比较：

### Separate

\[
J_{\mathrm{sep}}^{val}
=
L_{\mathrm{sep}}^{val}
+
\lambda_{\mathrm{local}}L_{\mathrm{part,sep}}^{val}
+
\lambda_GK.
\]

### Merged

\[
J_{\mathrm{merge}}^{val}
=
L_{\mathrm{merge}}^{val}
+
\lambda_{\mathrm{local}}L_{\mathrm{part,merge}}^{val}
+
\lambda_G(K-1).
\]

只有：

\[
\boxed{
J_{\mathrm{merge}}^{val}
<
J_{\mathrm{sep}}^{val}
}
\]

且至少：

```text
3/4 context folds
```

都支持 merge，才接受。

不允许因为 train error 下降就 merge。

## 15.4 合并后

保留一个槽位：

```text
winner = j
```

另一个：

```text
g_k = 0
status = dormant
```

它的参数保留，不物理删除。

---

# 16. \(\lambda_G\) 的选择

不能根据“最后是不是正好 3 个”来调 \(\lambda_G\)。

Development seed 0 只做一个小范围检查：

```text
lambda_G ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
```

记录：

```text
val NRMSE
alive candidate count
merge decisions
false merge on distinct GT mechanisms
```

选择原则：

\[
\boxed{
\text{在 validation prediction 几乎不变时优先更简单的模型。}
}
\]

具体 primary rule：

```text
从 val NRMSE 距离最优值不超过 2% 的设置中，
选择 alive candidate 最少的设置。
```

之后固定 \(\lambda_G\)，不得再根据 hidden GT recovery 调参。

## 16.1 B0 PASS

在 duplicate sanity test 上：

```text
C1/C4 被合并
C2/C3 不被错误合并
最终 alive = 3
IID NRMSE 相对 O5-B 恶化 < 5%
min functional R² 仍 > 0.90
```

若 B0 FAIL：

STOP。

说明“全局成本 + merge 判据”本身还不能可靠解决问题 B。

---

# 17. Adp2-B1：从随机 \(M_{\max}=5\) 开始做真正的全局收缩

只有：

```text
A3 有阶段性正结果
+
B0 PASS
```

以后才运行。

## 17.1 每轮顺序

```text
1. E-step：32 support exact search
2. 计算 assignment margin
3. unresolved sample 暂缓
4. 检查旧/new assignment 切换门槛
5. 更新 candidate evidence pool
6. 做跨 context reuse check
7. 小步 M-step
8. 更新 stable / provisional 状态
9. 对候选 pair 做 merge proposals
10. validation merge test
11. 接受必要 merge，更新 g_j
12. 记录全部结构指标
```

## 17.2 禁止直接按 usage 删除

不允许：

```text
usage < 0.1 → delete
```

因为真实但罕见 mechanism 可能 usage 很低。

candidate 只有在：

```text
长期低 evidence
+
无法跨 context 形成稳定 identity
+
被另一个 candidate 在 held-out merge test 中替代
```

时才能进入 dormant。

---

# 18. 为什么 Adp2 第一版不主动“拆分”

当前 E0 的主要失败是：

```text
一个真实 mechanism 被多个 candidate 片段化
```

而不是已经证明：

```text
一个 candidate 稳定吞掉多个真实 mechanisms
```

因此主实验先验证：

\[
\boxed{
\text{identity 稳定 + merge + global count}
}
\]

不要同时加入 split。

否则如果结果改变，无法知道是 merge 还是 split 起作用。

---

# 19. Adp2-S：只有出现过度合并时才进入拆分实验

触发条件：

```text
alive candidate < 3
或
prediction 仍较好但 functional recovery 显示一个 candidate 同时对应多个 GT
或
某个 stable candidate 的 cross-context reuse 长期很差
```

## 19.1 拆分思想

一个 candidate \(j\) 的 evidence pool：

\[
\mathcal E_j
\]

如果无法由同一个 \(\mathcal M_j\) 跨 context 统一解释，但分成两个子集后，两套 mechanism 在 held-out context 上显著更好，则提出 split。

## 19.2 第一版 split proposal

只对 candidate 的 residual target：

\[
r_{i,-j}
\]

做两组划分。

第一版使用：

```text
k-means(k=2)
```

仅作为 proposal generator，不作为最终决策。

分别训练临时：

\[
\mathcal M_{j_a},\quad
\mathcal M_{j_b}.
\]

比较：

### One candidate

\[
J_{\mathrm{one}}^{val}
\]

### Two candidates

\[
J_{\mathrm{two}}^{val}
+
\lambda_G
\]

因为 split 会多养一个 mechanism。

只有：

\[
\boxed{
J_{\mathrm{two}}^{val}+\lambda_G
<
J_{\mathrm{one}}^{val}
}
\]

且至少 3/4 context folds 支持，才 split。

## 19.3 dormant slot reuse

split 时优先复用 dormant slot。

如果没有 dormant slot：

```text
不允许超过 M_max
```

---

# 20. Adp2 完整伪代码

```text
Input:
    visible train / val data
    M_max = 5
    random mechanism bank
    all g_j = 1

Build fixed context groups from S only

for round = 0 ... R-1:

    # Step 1: exact local explanation search
    freeze all mechanisms

    for each sample i:
        enumerate all supports allowed by g
        optimize active v under each support

        get:
            best support
            second-best support
            J_best
            J_second

        margin_i = (J_second - J_best) / (abs(J_best) + eps)

        if margin_i < tau_conf:
            mark unresolved
            keep old stable assignment if available
        else:
            if old assignment exists:
                improvement = (J_old - J_best) / (abs(J_old) + eps)

                if improvement >= tau_switch:
                    accept new support
                else:
                    keep old support
            else:
                accept best support

    # Step 2: update evidence pools
    for candidate j:
        collect samples:
            resolved
            assignment stable >= 2 rounds
            m_ij = 1

        update evidence pool E_j

    # Step 3: cross-context reuse check
    for candidate j:
        if evidence is sufficient:
            evaluate reuse error across 4 context groups

            if:
                coverage >= 3 groups
                and reuse_error < tau_reuse
                for P_stable rounds:
                    mark stable
            else:
                    mark provisional

    # Step 4: conservative mechanism update
    for candidate j with g_j = 1:
        use only resolved/stable evidence

        fit residual target:
            r_i,-j = DeltaS - sum_{k!=j} e_i^k

        if stable:
            small learning rate / few steps
        else:
            normal Adp2 learning rate / steps

    # Step 5: candidate consolidation
    if Adp2-B1 enabled:
        generate pairwise merge proposals

        for pair (j,k):
            train temporary merged candidate on train evidence
            evaluate separate vs merged on held-out context / val

            if merged global objective lower
               in >= 3/4 folds:
                accept merge
                set one slot dormant

    # Step 6: logging
    evaluate:
        reconstruction
        assignment stability
        candidate usage
        reuse
        alive count
        functional recovery (evaluator only)
        participation recovery (evaluator only)

    save checkpoint + assignments
```

---

# 21. Development 数据规模

Adp2 的 reuse/merge 检查比 Adp1 更贵。

不要一开始直接使用 100k。

## 21.1 开发阶段

```text
train = 4,096
val = 1,024
test_iid = 2,048
test_context = 2,048
test_101 = 2,048
seed = 0
rounds = 20
```

原因：

Adp1 已证明 1,024 可能过小，而 4,096 足够观察结构动力学，同时成本可控。

## 21.2 中等规模确认

只有 development 有正结果后：

```text
train = 16,384
val = 4,096
test_iid = 4,096
test_context = 4,096
test_101 = 4,096
optimization seeds = 0,1,2
```

## 21.3 正式确认

只有中等规模至少：

```text
2/3 seeds
```

达到完整结构恢复趋势后，再进入：

```text
原 E0 full dataset
train = 100,000
val = 20,000
test_iid = 20,000
test_context = 20,000
test_101 = 20,000
optimization seeds = 0..9
```

如果 exact search 太慢，可以先只做 10 seeds 的 16,384 规模确认。

不要因为计算量大而提前把 \(q_\eta\) 加回来。

Adp2 首先验证 discovery dynamics，不验证快速 inference。

---

# 22. 关键指标及其含义

## 22.1 Prediction NRMSE

继续使用：

\[
NRMSE_W
=
\frac{
\sqrt{E\|\Delta S-\widehat{\Delta S}\|^2}
}{
\sqrt{E\|\Delta S-E[\Delta S]\|^2}
}.
\]

报告：

```text
train
val
test_iid
test_context
test_101
```

只说明 prediction，不证明 mechanism recovery。

## 22.2 Participation recovery

Hungarian matching 后：

```text
precision
recall
F1
AUROC
```

最终目标：

\[
F1>0.90.
\]

## 22.3 Instance-effect \(R^2\)

比较：

\[
\hat e_i^j
\]

与 GT：

\[
e_i^{j*}.
\]

Hungarian matching 后报告：

```text
matrix
matched
minimum_matched
```

目标：

\[
\min R^2>0.90.
\]

## 22.4 Functional \(R^2\)

继续沿用 E0 的 functional evaluation。

目标：

\[
\min R^2>0.90.
\]

这是判断 candidate 是否真的学到可复用 mechanism function 的核心指标。

## 22.5 Assignment 稳定性

同时报告：

### bit change

\[
SCR
=
\frac1{NM_{\max}}
\sum_{i,j}
\mathbf1[m_{ij}^{(r)}\neq m_{ij}^{(r-1)}].
\]

### exact agreement

\[
A^{(r)}
=
\frac1N
\sum_i
\mathbf1[m_i^{(r)}=m_i^{(r-1)}].
\]

最终目标：

```text
last-3-round mean SCR < 0.01
```

## 22.6 Candidate population usage

\[
u_j
=
\frac1N\sum_i m_i^j.
\]

注意：

```text
usage 低 ≠ automatically redundant
usage 高 ≠ automatically real mechanism
```

只作为结构观察量。

## 22.7 Context reuse error

使用 A0 定义：

\[
E_j^{reuse}.
\]

必须同时报告：

```text
reuse error
context coverage
samples per group
```

低 evidence 不允许被解释成 mechanism 不存在。

## 22.8 Alive mechanism count

\[
K_{\mathrm{alive}}
=
\sum_jg_j.
\]

learner 不知道：

\[
M_{\mathrm{real}}=3.
\]

只有 evaluator 最终比较：

\[
K_{\mathrm{alive}}
\]

与 GT。

## 22.9 Merge quality

每次 merge 保存：

```text
pair
separate val objective
merged val objective
context-fold votes
prediction change
candidate count change
```

evaluator 额外报告：

```text
true duplicate merge
false merge between distinct GT mechanisms
```

---

# 23. 101 split 的特殊评价

GT：

\[
m=(1,0,1).
\]

不要对恒为 0 的 \(e_2^*\) 计算 \(R^2\)。

报告：

```text
101 NRMSE
exact participation pattern accuracy
Z1 effect R²
Z3 effect R²
inactive-Z2 false-positive rate
E||predicted e2||
```

目标：

```text
101 NRMSE < 0.10
inactive Z2 FPR < 0.10
Z1/Z3 effect R² > 0.90
```

---

# 24. 最终 PASS / FAIL 标准

Adp2-F 单 seed 完整 PASS：

| 项目 | 标准 |
|---|---:|
| IID NRMSE | < 0.05 |
| 101 NRMSE | < 0.10 |
| Participation mean F1 | > 0.90 |
| min matched instance-effect \(R^2\) | > 0.90 |
| min matched functional \(R^2\) | > 0.90 |
| last-3-round mean SCR | < 0.01 |
| redundant candidate usage | < 0.10 |
| alive candidate count | 3 |
| false merge | 0 |

正式多 seed：

\[
\boxed{RecoveryRate\ge 8/10}
\]

才视为 strong pass。

---

# 25. 分阶段结论规则

## 情况 1：A0 FAIL

说明：

\[
\boxed{
\text{我们当前无法从 observable residual 中可靠测量“跨 context 可复用”。}
}
\]

不要进入 A3/B1。

需要先重新设计 identity evidence。

## 情况 2：A1 明显改善，A2 再改善

说明：

\[
\boxed{
\text{Adp1 的主要问题确实包含低证据 pseudo-assignment 与频繁切换。}
}
\]

继续 A3。

## 情况 3：A1/A2 让 SCR 下降，但 functional \(R^2\) 仍差

说明：

\[
\boxed{
\text{只是把错误 assignment 冻住了。}
}
\]

问题不是稳定性，而是缺少真正的 mechanism identity evidence。

A3 成为关键。

## 情况 4：A3 出现三个高 reuse candidate，但五个 candidate 都仍 alive

说明：

\[
\boxed{
\text{问题 A 基本解决，问题 B 仍在。}
}
\]

进入 B0/B1。

## 情况 5：B0 PASS，B1 FAIL

说明：

\[
\boxed{
\text{merge + global cost 在“重复已知机制”时有效，}
\text{但 random discovery 仍无法先形成可合并的局部机制。}
}
\]

真正瓶颈仍是 identity formation，而不是 model-order selection。

## 情况 6：B1 过度合并到 1--2 个 candidate

说明：

```text
global cost 过强
或
reuse / merge 判据把不同 mechanisms 当成同一种
```

先检查 B0 false merge，再考虑 Adp2-S。

## 情况 7：B1 稳定到 3 个 candidate，但 functional \(R^2\) 仍差

说明：

\[
\boxed{
\text{“数量正确”不等于“机制正确”。}
}
\]

不能宣称成功。

## 情况 8：B1 达到 3 个 candidate + functional recovery 高 + 101 泛化好

这才支持：

\[
\boxed{
\text{局部 transformation fragments 可以经过复用验证和全局整理，}
\text{形成正确数量的 reusable }Z_D.
}
\]

---

# 26. 必须做的消融对照

最终至少比较：

| 方法 | 低证据暂缓 | 切换门槛 | 跨 context 复用 | 全局机制成本 | merge |
|---|---:|---:|---:|---:|---:|
| Adp1 | × | × | × | × | × |
| Adp2-A1 | ✓ | × | × | × | × |
| Adp2-A2 | ✓ | ✓ | × | × | × |
| Adp2-A3 | ✓ | ✓ | ✓ | × | × |
| Adp2-B1 | ✓ | ✓ | ✓ | ✓ | ✓ |

最终表必须同时报告：

```text
IID NRMSE
101 NRMSE
F1
min instance R²
min functional R²
SCR
alive count
candidate usages
RecoveryRate
```

---

# 27. Negative controls

## NC-A：随机打乱 context group

保持 group size 不变，但随机打乱每个 sample 的 context id。

目的：

如果“跨 context 复用”真的依赖 state context，那么随机分组后的指标辨别力应下降。

## NC-B：去掉 evidence filtering

直接把所有 top-1 assignment 放入 M-step。

预期回到 Adp1-like oscillation。

## NC-C：去掉 global mechanism cost

保留 merge 检验，但令：

\[
\lambda_G=0.
\]

观察是否仍有动力把重复 candidate 合并。

## NC-D：只用 global mechanism cost，不做 reuse 验证

目的：

验证简单“少养几个 candidate”是否会导致错误合并。

如果 candidate 数量变成 3，但 functional recovery 仍差，则说明：

\[
\boxed{
\text{全局稀疏只解决数量，不解决 identity。}
}
\]

---

# 28. 工程目录建议

```text
outputs/
└── e0_adp2/
    ├── configs/
    │   ├── adp2_a0.json
    │   ├── adp2_a1.json
    │   ├── adp2_a2.json
    │   ├── adp2_a3.json
    │   ├── adp2_b0.json
    │   ├── adp2_b1.json
    │   └── adp2_full.json
    │
    ├── a0_reuse_calibration/
    │
    ├── a1_seed_0/
    │   ├── round_metrics.jsonl
    │   ├── checkpoints/
    │   ├── assignments/
    │   └── final_metrics.json
    │
    ├── a2_seed_0/
    ├── a3_seed_0/
    ├── b0_duplicate_sanity/
    ├── b1_seed_0/
    ├── negatives/
    └── full/
        ├── seed_0/
        ├── seed_1/
        └── ...
```

---

# 29. 推荐命令模板

按实际代码入口调整文件名，但保留实验编号。

```bash
# 0. Reproduce Adp1 with full telemetry
python run_adp2.py --stage adp2_0 --seed 0

# A0. Calibrate reuse metric
python run_adp2.py --stage adp2_a0 --seed 0

# A1. Evidence filtering only
python run_adp2.py --stage adp2_a1 --seed 0

# A2. + switch threshold
python run_adp2.py --stage adp2_a2 --seed 0

# A3. + cross-context reuse
python run_adp2.py --stage adp2_a3 --seed 0

# B0. Controlled duplicate merge sanity
python run_adp2.py --stage adp2_b0 --seed 0

# B1. Full random-init consolidation
python run_adp2.py --stage adp2_b1 --seed 0

# Medium-scale confirmation
python run_adp2.py --stage adp2_b1 --train-size 16384 --seed 0
python run_adp2.py --stage adp2_b1 --train-size 16384 --seed 1
python run_adp2.py --stage adp2_b1 --train-size 16384 --seed 2
```

---

# 30. 每次运行前的复现检查

必须保存：

```text
git commit
python executable
python version
torch version
CUDA available
GPU model
world seed
dataset seed
optimization seed
context seed
config hash
```

特别检查：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

Adp1 的 16,384 诊断曾出现 CPU build 与原 CUDA 环境不一致的问题。

Adp2 不允许再把不同运行环境的 wall-clock 直接比较。

---

# 31. 每次运行后必须生成的最终报告

每个 stage 自动生成：

```text
final_metrics.json
summary.md
```

`summary.md` 至少包含：

```text
1. 实验设置
2. 相对上一 stage 唯一新增机制
3. 最终预测指标
4. 最终结构指标
5. 每轮 SCR / exact agreement
6. candidate usage trajectories
7. reuse trajectories
8. alive candidate count trajectories
9. merge/split event log
10. 101 special evaluation
11. PASS/FAIL
12. 失败类型
```

---

# 32. 不允许用来“证明成功”的现象

以下任何一项单独出现都不能算 Adp2 成功：

```text
reconstruction loss 下降
IID NRMSE 比 Adp1 低
平均每个 sample active 数接近 3
alive candidate 数恰好等于 3
candidate usage 看起来不均匀
SCR 很低
```

真正成功要求同时：

\[
\boxed{
\text{正确数量}
+
\text{稳定 identity}
+
\text{functional recovery}
+
\text{组合泛化}
}
\]

---

# 33. Adp2 最终要回答的科学问题

Adp2 不是为了证明某个训练技巧有效。

它要验证一个更重要的认识：

\[
\boxed{
\text{有效 mechanism 的 identity 是否可以由“跨 interaction 的可复用性”逐渐建立？}
}
\]

以及：

\[
\boxed{
\text{当 candidate identity 已经具有证据后，}
\text{是否可以通过全局模型复杂度比较，在不知道 }M_{\mathrm{real}}\text{ 的情况下自动整理出正确数量的 }Z_D？
}
\]

如果 Adp2-F 最终通过，则我们的 \(TP\rightarrow Z_D\) 发现逻辑将从：

```text
一次 reconstruction
→ 稀疏 candidate
```

更新为：

```text
局部 transformation 片段
→ 暂定解释
→ 跨 context 重复验证
→ 稳定 mechanism identity
→ 合并重复 mechanism
→ 必要时拆分混合 mechanism
→ 得到最小可复用 Z_D 集合
```

如果 Adp2 仍失败，尤其是在：

```text
A0 metric 有辨别力
A1/A2 assignment 已稳定
A3 仍不能形成 GT-like mechanisms
```

的情况下，就有较强证据说明：

\[
\boxed{
\text{仅靠被动观察到的 transformation repetition 仍不足以定义 mechanism identity。}
}
\]

下一步才应该认真引入：

```text
主动 intervention
intervention provenance
机制等价实验
主动选择能够区分 candidate hypotheses 的动作
```

而不是继续增加无结构依据的 regularizer。

---

# 34. 最终执行顺序

工程上严格按以下顺序：

```text
STEP 0  补日志 + 复现 Adp1
STEP 1  Adp2-A0：验证 reuse metric
STEP 2  Adp2-A1：低证据 assignment 暂缓
STEP 3  Adp2-A2：加入切换门槛
STEP 4  Adp2-A3：加入跨 context 复用确认
STEP 5  Adp2-B0：重复 mechanism 合并 sanity
STEP 6  固定 lambda_G
STEP 7  Adp2-B1：random-init full consolidation
STEP 8  只有过度合并时运行 Adp2-S
STEP 9  中等规模 3 seeds
STEP 10 正式 10 seeds
```

在 STEP 7 之前：

```text
不要运行 relation
不要进入 E1
不要引入 q_eta amortization
不要加入视觉
不要修改 synthetic generator
```

---

# 35. 一句话判定标准

Adp2 真正想看到的不是：

\[
5\rightarrow3
\]

而是：

\[
\boxed{
\text{5 个没有身份的 candidate}
\rightarrow
\text{若干局部 transformation fragments}
\rightarrow
\text{跨 context 被反复验证的稳定机制}
\rightarrow
\text{合并重复、保留不同}
\rightarrow
\text{3 个与 GT mechanism 功能对应的 }Z_D.
}
\]

只有最后一步同时满足原 E0 的 prediction、participation、functional recovery 和 101 composition 标准，才算真正支持我们的想法。
