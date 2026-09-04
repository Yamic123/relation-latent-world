# Adp4 实验执行方案：从局部可比 TP 形成局部作用单元，并对齐为全局 \(Z_D\)

> **核心目标**
>
> Adp4 不再把单个 fragment 独立分配给 candidate，而是显式利用 TP 之间的**局部可比结构**：
>
> \[
> \boxed{
> \text{可比 TP}
> \rightarrow
> \text{局部作用单元}
> \rightarrow
> \text{组内联合匹配}
> \rightarrow
> \text{跨组功能对齐}
> \rightarrow
> \text{合并 realization 碎片}
> \rightarrow
> Z_D
> }
> \]
>
> 同时，Adp4 必须验证这一做法能否从 E0 中的“oracle 可比关系”逐步过渡到未来整体架构中的“自动发现哪些 TP 值得比较”。

---

# 0. Adp4 的出发点

Adp3 已经证明两件事：

1. clean single-change fragment 确实显著降低了机制发现难度，至少一个 GT-like mechanism 可以形成；
2. 逐 fragment 独立 hard assignment 仍会把同一个高层 \(Z_D\) 按低层 realization \(v\) 的不同区域拆给多个 candidate。

额外诊断进一步表明：

- 同一 base、同一真实机制对应的 4 条 contrast 在当前 \(R_D=0\) 数据中几乎完全相同；
- 但 Adp3-A1 会把这 4 条重复证据分给不同 candidate；
- 训练后期这种拆分并未自然消失；
- 典型失败是同一个真实 mechanism 在 \(v<0\) 与 \(v>0\) 区域被不同 candidate 分担。

因此 Adp4 的核心修正不是增加新的正则，而是改变**学习单位与匹配层次**：

\[
\boxed{
\text{fragment 不再是最小身份单位；}
\quad
\text{同一局部条件下重复支持同一种作用的 fragments 先形成 local unit。}
}
\]

---

# 1. 与整体架构的对齐原则

Adp4 中必须严格区分两件事：

## 1.1 \(Z_D\) 的定义

\(Z_D\) 仍然是领域级高层有效机制：

- 跨 TP 复用；
- 跨时间一致；
- 未来应跨任务复用；
- 低层状态、具体 realization 可以变化；
- 与其它 \(Z_D\) 的系统性相互影响未来由 \(R_D\) 表示。

所以：

\[
\boxed{
\text{“组内匹配”不是 }Z_D\text{ 的定义。}
}
\]

## 1.2 组内匹配的角色

组内匹配只是为了找到一批：

\[
\boxed{
\text{在当前证据下足够可比较的 TP}
}
\]

从而得到更干净的局部作用证据。

最终架构中不应依赖永久的人工 `group_id`。

因此 Adp4 分两大阶段：

```text
A：oracle / 强可比关系
   先验证局部作用单元 + 组内联合匹配算法本身是否有效

B：自动发现可比关系
   去掉 oracle group/pair 信息，只使用 learner 可见的 S 与干预 A 建图
```

只有 A 成功后才进入 B。

---

# 2. 当前 E0 条件

保持原 E0-A world：

```text
WORLD_SEED = 20260901
DATASET_SEED = 20260902
M_real = 3           # evaluator only
M_max = 5
state slots = 3
slot dim = 2
v dim = 1
TP = identity
R_D = 0
mechanism architecture = 原 SetMechanismBank
```

真实世界仍满足：

\[
\Delta S_i
=
\sum_{j=1}^{3}
m_i^j\mathcal M_j^*(S_i,v_i^j).
\]

禁止修改：

```text
A_jk, b_jk, c_jk
state distribution
v distribution
GT mechanism functions
heldout 101 definition
mechanism network capacity
```

---

# 3. Adp4 总体执行顺序

```text
Adp4-0   复核现有 Adp3 数据中的 local-unit 结构
   ↓
Adp4-A0  不用 GT label，从同一 base 的 fragments 自动形成 3 个 local units
   ↓
Adp4-A1  local-unit 级组内联合匹配；禁止逐 fragment 独立 assignment
   ↓
Adp4-A2  跨 realization 的功能合并，解决同一 Z_D 被正/负 v 等区域拆开
   ↓
Adp4-A3  用学出的 Z_D 回到完整 TP 推断 m,v
   ↓
Adp4-B0  去掉显式 group_id，仅从 S 自动恢复“局部可比邻域”
   ↓
Adp4-B1  加入 learner-visible 干预 A，自动发现 single-change TP edges
   ↓
Adp4-B2  完整自动 comparison graph → local units → Z_D
   ↓
Adp4-F   多 seed 确认
```

任何一步 FAIL，先 STOP 并解释，不自动进入下一阶段。

---

# 4. Adp4-0：只读复核当前 Adp3 数据

不训练。

对每个 base group：

- 12 个 contrast fragments；
- hidden GT 仅 evaluator 使用。

验证：

1. 每个 base 是否恰好可以分成 3 个真实作用 family；
2. 每个 family 是否有 4 条重复 contrast；
3. 同 family 四条 \(d\) 的最大差异；
4. 不同 family 之间 \(d\) 的最小距离。

输出：

```text
local_unit_sanity.json
within_family_max_distance
between_family_min_distance
separation_ratio
```

### PASS

当前 E0 数值精度下：

```text
within_family_max_distance << between_family_min_distance
```

并且所有 base 均为 3×4 结构。

---

# 5. Adp4-A0：从 observable fragment 自动形成 local units

这里允许使用：

```text
group_id
S
d
```

但禁止使用：

```text
GT mechanism index
GT m
GT v
GT e
```

## 5.1 在每个 base 内构造 fragment 距离

对于同一 base 的 fragments \(a,b\)：

\[
D_{ab}^{frag}
=
\frac{
\|d_a-d_b\|_2
}{
\frac12(\|d_a\|_2+\|d_b\|_2)+\epsilon
}.
\]

第一版不学习 embedding，直接比较真实 transformation difference。

## 5.2 形成 local units

由于当前 E0-A 中同一局部作用的四条 contrast 理论上相同，使用固定小阈值：

\[
D_{ab}^{frag}<\tau_{unit}.
\]

\(\tau_{unit}\) 不能根据 GT cluster accuracy 调。

取：

```text
tau_unit = max(
    1e-6,
    10 × train 内数值重复误差的 99.9% 分位数
)
```

对 fragment graph 求连通分量。

每个 base 预期得到：

\[
U_g=\{u_g^1,u_g^2,u_g^3\}.
\]

每个 local unit 保存：

```text
base_id
S_g
member_fragment_ids
d_mean
d_variance
```

## 5.3 evaluator 指标

GT 只用于评估：

```text
number of units/base
unit purity
unit completeness
```

### A0 PASS

```text
>= 99% base 得到恰好 3 个 units
mean unit purity > 0.99
mean unit completeness > 0.99
```

若 FAIL：

STOP。

---

# 6. local unit 的训练表示

每个 local unit：

\[
u=(S,\{d_r\}_{r=1}^{n_u})
\]

只对应一个局部作用。

它只允许有：

- 一个 global candidate assignment \(c_u\)；
- 一个共享 realization \(v_u\)。

禁止出现：

\[
d_1\to C_1,\quad
d_2\to C_3
\]

这种同一 local unit 内部的拆分。

---

# 7. Candidate 对 local unit 的解释代价

对于 local unit \(u\) 和 candidate \(j\)：

\[
E_{uj}
=
\min_v
\frac1{|u|}
\sum_{r\in u}
D_S
\left(
d_r,
\mathcal M_j(S_u,v)
\right).
\]

第一版保持 Adp3 的 realization 优化：

```text
v = 1.5 tanh(z)
inner optimizer = Adam
inner lr = 0.05
inner steps = 20
restarts = 2
```

但同一个 local unit 的所有成员共享同一个 \(v_u\)。

---

# 8. Adp4-A1：组内联合匹配

这是 Adp4 的核心实验。

每个 base 有三个 local units：

\[
U_g=\{u_g^1,u_g^2,u_g^3\}.
\]

维护：

\[
M_{\max}=5
\]

个 global candidates：

\[
\mathcal M_1,\ldots,\mathcal M_5.
\]

## 8.1 禁止逐 unit 独立 argmin

不能分别：

\[
c_{g,a}
=
\arg\min_j E_{gaj}.
\]

因为同一 base 中三个 local units 是三个不同的局部作用。

## 8.2 对整个 base 做一次联合匹配

求：

\[
\boxed{
\pi_g^*
=
\arg\min_{\pi_g}
\sum_{a=1}^{3}
E_{g,a,\pi_g(a)}
}
\]

满足：

\[
\pi_g(a)\neq\pi_g(b),
\qquad a\neq b.
\]

即同一个 base 的三个 local units 必须匹配到三个不同的 global candidate。

由于只有 3 units、5 candidates，可直接枚举：

\[
P(5,3)=60
\]

种 injective assignment，无需近似。

## 8.3 M-step

固定所有：

\[
\pi_g,\ v_{g,a}.
\]

对 candidate \(j\)：

\[
\mathcal D_j
=
\{
(g,a):\pi_g(a)=j
\}.
\]

训练：

\[
\mathcal M_j(S_g,v_{g,a})
\approx
d_{g,a}.
\]

配置：

```text
optimizer = AdamW
lr = 3e-4
weight_decay = 1e-5
batch_size = 128 local units
epochs_per_round = 4
rounds = 40
```

## 8.4 A1 暂时不加入

```text
global count cost
merge
split
TP reconstruction
full-TP participation
```

只回答：

\[
\boxed{
\text{local unit + base 内联合匹配是否足以显著改善 mechanism identity formation？}
}
\]

---

# 9. A1 必须记录的指标

每轮保存：

```text
train local-unit MSE
val local-unit NRMSE

candidate usage[5]
candidate unit_count[5]

base assignment change rate
exact base assignment agreement

functional R2 matrix [5 x 3]
matched functional R2

unit Hungarian accuracy
unit contingency matrix

candidate state coverage
candidate effect-norm range

hidden evaluator:
candidate GT-family purity
candidate v-sign distribution
candidate |v| quantiles
```

最后三个仅 evaluator 可见，不参与训练。

---

# 10. A1 对“按 v 拆分”的专门诊断

对每个 GT mechanism \(k\) 和 learned candidate \(j\)，只在 evaluator 中统计：

\[
P(C_j\mid Z_k,v<0)
\]

与：

\[
P(C_j\mid Z_k,v>0).
\]

定义 sign-split 指标：

\[
Split_k
=
\frac12
\sum_j
\left|
P(C_j\mid Z_k,v<0)
-
P(C_j\mid Z_k,v>0)
\right|.
\]

越接近 0，说明同一个高层机制没有因为 realization 正负号被拆开。

同时对 \(|v|\) 分四个分位区间做同样统计。

---

# 11. A1 阶段判定

### 强正向结果

相比 Adp3-A1：

- assignment change 显著下降；
- 三个 GT family 均出现至少一个明显对应 candidate；
- sign-split 明显减弱；
- functional \(R^2\) 不再只有一个 family 成形。

### A1 PASS

development seed 0：

```text
unit Hungarian accuracy > 0.80
三个 GT 各自 best functional R² > 0.70
其中至少两个 > 0.85
last-3 base assignment change < 0.15
mean sign-split < 0.25
```

这里暂不要求只有 3 个 alive candidates。

如果出现 3 个真实 family 已形成，但 4/5 个 candidate 是同 family 的 realization 子区间，则进入 A2。

---

# 12. Adp4-A2：合并 realization 碎片

A2 解决的问题不是“identity 从无到有”，而是：

\[
\boxed{
\text{同一个 }Z_D
\text{ 被多个 candidate 按 }v\text{ 或状态区域拆开。}
}
\]

## 12.1 不按 usage 直接删 candidate

禁止：

```text
usage < threshold → delete
```

候选是否重复必须由**功能可合并性**判断。

## 12.2 pairwise union test

对 candidates \(j,k\)：

训练临时共同机制：

\[
\mathcal M_{j\oplus k}
\]

使用：

\[
\mathcal D_j\cup\mathcal D_k.
\]

每个 local unit 仍可重新优化自己的 \(v\)。

比较：

### Separate

\[
L_{\rm sep}^{val}
\]

### Merged

\[
L_{\rm merge}^{val}.
\]

## 12.3 验证集必须同时覆盖两侧 realization

为了专门检测“负 \(v\) 一个 candidate、正 \(v\) 另一个 candidate”的情况，validation split 按以下方式分层：

```text
source candidate j/k
state context
effect-norm quartile
```

注意训练决策不能使用 GT v 的正负。

GT v-sign 只做 evaluator。

## 12.4 全局模型成本

维护：

\[
g_j\in\{0,1\}.
\]

全局目标：

\[
J_{\rm global}
=
L_{\rm unit}^{val}
+
\lambda_G\sum_jg_j.
\]

接受 merge 当：

\[
L_{\rm merge}^{val}
-
L_{\rm sep}^{val}
<
\lambda_G
\]

且至少：

```text
3/4 state-context folds 支持 merge
```

## 12.5 \(\lambda_G\) 选择

development seed 0：

```text
lambda_G ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
```

选择规则：

```text
val local-unit NRMSE 距最优 <= 2% 的配置中，
选择 alive candidate 最少者。
```

不能根据“是不是正好 3 个”调参。

---

# 13. A2 PASS

要求：

```text
alive candidate count = 3
三个 matched functional R² > 0.90
unit Hungarian accuracy > 0.90
false merge = 0
mean sign-split < 0.10
```

若 count=3 但 functional recovery 不好，不算 PASS。

---

# 14. Adp4-A3：回到完整 TP

只有 A2 PASS 后执行。

冻结：

\[
\mathcal M_1,\mathcal M_2,\mathcal M_3.
\]

回到原始 E0 full TP：

\[
(S_i,p_i,\Delta S_i).
\]

枚举：

\[
2^3=8
\]

个 support，并优化 realization：

\[
(m_i,v_i)
=
\arg\min
\left[
D_S
\left(
\Delta S_i,
\sum_jm_i^j\mathcal M_j(S_i,v_i^j)
\right)
+
\lambda_P\|m_i\|_0
\right].
\]

## A3 PASS

```text
IID NRMSE < 0.05
101 NRMSE < 0.10
participation F1 > 0.90
min instance-effect R² > 0.90
min functional R² > 0.90
inactive Z2 FPR on 101 < 0.10
```

这一步回答：

\[
\boxed{
\text{先形成高层 mechanism identity 后，完整 TP participation 是否可恢复？}
}
\]

---

# 15. 到这里为止，Adp4-A 仍使用强可比关系

A0-A3 的目的只是验证：

\[
\boxed{
\text{如果 learner 得到了正确的“哪些 TP 值得局部比较”的关系，
local-unit → global-Z_D 这套学习方式是否成立。}
}
\]

它不是最终系统。

因此 A 成功后必须继续 B。

---

# 16. Adp4-B0：去掉显式 group_id

当前 contrast 数据中，同一 base 的 TP/fragment 共享完全相同的：

\[
S^{pre}.
\]

B0 删除 learner-facing `group_id`。

只根据：

\[
S^{pre}
\]

建立局部邻域。

定义：

\[
D_S(i,j)
=
\|S_i^{pre}-S_j^{pre}\|_2.
\]

当前 E0 第一版使用：

```text
tau_state = 1e-8
```

构造 connected components。

这些 component 替代 oracle group_id。

## B0 PASS

要求：

```text
自动恢复 group 的 pairwise F1 > 0.99
```

然后完整重复 A0-A2。

如果结果与 oracle group 版本近似一致，说明：

\[
\boxed{
\text{算法不依赖人工 group_id 本身。}
}
\]

---

# 17. 但 B0 仍然不是最终架构

真实系统不会反复遇到完全相同的 \(S^{pre}\)。

同时最终 TP 中存在：

\[
A_i
\]

这一 learner-visible 干预信息。

因此 B1 引入一个最小 bridge dataset，用于测试：

\[
\boxed{
\text{能否从状态 + 低层干预本身发现哪些 TP 可比较。}
}
\]

---

# 18. Adp4-B1：learner-visible 干预 \(A\)

这一阶段明确标记为：

\[
\boxed{\text{E0-to-full-architecture bridge}}
\]

不是原 E0-A 的纯条件。

## 18.1 hidden control

仅 generator 内部定义：

\[
c_m
=
(m_1v_1,m_2v_2,m_3v_3)^\top.
\]

## 18.2 learner-visible action

固定随机正交矩阵：

\[
Q_A\in\mathbb R^{3\times3},
\]

定义：

\[
\boxed{
A_m=Q_Ac_m.
}
\]

learner 只能看到 \(A_m\)，看不到：

```text
m
v
c
Q_A^{-1}
```

并且：

\[
\boxed{
A\text{ 只允许用于“哪些 TP 可比较”的发现，不输入 }\mathcal M_j.
}
\]

这样避免把 action 直接当作 \(Z_D\)。

---

# 19. 自动 TP comparison graph

从 raw TP：

\[
\mathcal I_i=(S_i^{pre},A_i,S_i^{post})
\]

构图。

## 19.1 节点

每个 TP 一个节点。

## 19.2 先找 state-local neighborhood

如果：

\[
D_S(S_i^{pre},S_j^{pre})<\tau_S
\]

则进入同一个候选 comparison neighborhood。

E0 bridge 第一版仍使用相同 base state，后续 robustness 再加入小扰动。

## 19.3 在 neighborhood 内比较 action difference

\[
\delta A_{ij}=A_j-A_i.
\]

对于同一个 base 的 8 个 intervention，真正的 single-factor edge 会产生三类反复出现的 action-difference direction。

将：

\[
\frac{\delta A_{ij}}{\|\delta A_{ij}\|}
\]

按方向聚类，允许整体正负号等价。

只保留：

```text
在同一 state neighborhood 中重复出现 >= 3 次
```

的方向族作为候选 single-change relation。

不使用 GT Hamming distance。

---

# 20. B1 的 pair discovery evaluator

GT 只评价：

```text
edge precision
edge recall
edge F1
```

目标是真正 Hamming-distance-1 的 TP pair。

### B1 PASS

```text
single-change edge precision > 0.95
single-change edge recall > 0.90
```

如果 FAIL：

STOP。

不要把错误 edge 输入 Z_D learner。

---

# 21. Adp4-B2：完全由 comparison graph 产生 local units

删除 learner-facing：

```text
group_id
single-change pair label
GT mechanism label
```

learner 只看到：

```text
S_pre
A
S_post / DeltaS / TP
```

流程：

```text
1. 根据 S 找局部 state neighborhood
2. 根据重复 action-difference direction 找可比较 TP edges
3. edge 上计算 transformation difference d
4. 将同一 neighborhood 中重复支持同一 d 的 edges 聚成 local unit
5. 每个 neighborhood 内对 local units 做 injective matching
6. 跨 neighborhood 更新 global mechanism candidates
7. 做 functional merge
8. 得到 Z_D
```

这才是与未来整体架构直接对齐的版本。

---

# 22. B2 PASS

相对于 A2：

```text
alive count = 3
matched functional R² each > 0.85
unit Hungarian accuracy > 0.85
full TP participation F1 > 0.85
101 NRMSE < 0.12
```

B2 允许比 oracle-comparability A2 稍低，但必须明显高于 Adp3-A1。

---

# 23. Robustness：近似可比，而非完全相同状态

只有 B2 PASS 后运行。

对每个 base 的重复 interaction 加小状态扰动：

\[
S_{g,r}
=
S_g+\epsilon_{g,r},
\qquad
\epsilon\sim\mathcal N(0,\sigma_S^2I).
\]

测试：

```text
sigma_S ∈ {0.01, 0.03, 0.05}
```

此时不再要求：

\[
d_a=d_b.
\]

local-unit 判断改为：

> 一组 edge 是否能被同一个局部机制函数在近邻状态上解释。

这一阶段才开始接近真实连续经验流。

---

# 24. 关于 \(R_D\)

Adp4 全部核心实验仍使用：

\[
R_D=0.
\]

因此 local contrast 可以近似解释为单一 \(Z_D\) 作用。

**不要在 Adp4 中同时加入 relation。**

Adp4 PASS 后的下一阶段才测试：

\[
R_D\neq0
\]

时：

- local-unit 证据是否仍能形成初步 \(Z_D\)；
- 多机制共现的系统残差能否稳定归入 \(R_D\)；
- 是否能够避免 \(Z_D\) 吞掉 relation effect。

---

# 25. 必须做的消融

| 方法 | local unit | 组内 injective matching | functional merge | 自动 comparison graph |
|---|---:|---:|---:|---:|
| Adp3-A1 | × | × | × | × |
| Adp4-A0/A1 | ✓ | ✓ | × | × |
| Adp4-A2 | ✓ | ✓ | ✓ | × |
| Adp4-B0 | ✓ | ✓ | ✓ | state-only |
| Adp4-B2 | ✓ | ✓ | ✓ | state + action |

核心要回答：

1. local unit 是否比独立 fragment 更好；
2. injective matching 是否减少同一 base 内错误混合；
3. merge 是否消除 realization 分裂；
4. oracle comparability 能否被 learner 自己发现的 comparison graph 替代。

---

# 26. 负对照

## NC-1：取消 unit shared assignment

同一 local unit 的 4 条 fragment 重新独立 assignment。

预期回到 Adp3-like fragmentation。

## NC-2：取消 base 内 injective constraint

三个 local units 可分给同一个 candidate。

检查 functional identity 是否恶化。

## NC-3：打乱 comparison edges

保持 edge 数量不变，但随机连接不同 TP。

预期 local-unit purity 和 functional recovery 显著下降。

## NC-4：把 action \(A\) 直接输入 mechanism

这是禁止的主方案，但可作为负对照：

\[
\mathcal M_j(S,v,A).
\]

如果 prediction 好但 ZD functional identity 变差，说明 action shortcut 会破坏高层 mechanism discovery。

---

# 27. 每阶段输出

目录：

```text
outputs/
└── e0_adp4/
    ├── adp4_0/
    ├── a0_local_units/
    ├── a1_group_matching/
    ├── a2_functional_merge/
    ├── a3_full_tp/
    ├── b0_auto_state_groups/
    ├── b1_pair_discovery/
    ├── b2_auto_graph/
    └── robustness/
```

每阶段保存：

```text
config.json
final_metrics.json
summary.md
round_metrics.jsonl
checkpoints/
assignments/
```

---

# 28. 每轮核心 telemetry

```text
round

train_unit_mse
val_unit_nrmse

candidate_usage
candidate_unit_count

base_assignment_change_rate
base_exact_agreement

unit_matching_accuracy
unit_contingency

functional_r2_matrix
matched_functional_r2

alive
num_alive

merge_proposals
merge_accepts

comparison_graph_precision   # B only
comparison_graph_recall      # B only
comparison_graph_f1          # B only

runtime
```

evaluator-only：

```text
GT-family purity
v-sign split index
v-magnitude split
false merge
```

---

# 29. 推荐执行命令

```bash
# current Adp3 data sanity
python run_adp4.py --stage adp4_0 --seed 0

# local-unit construction
python run_adp4.py --stage adp4_a0 --seed 0

# group-wise injective assignment
python run_adp4.py --stage adp4_a1 --seed 0

# functional merge
python run_adp4.py --stage adp4_a2 --seed 0

# return to full TP
python run_adp4.py --stage adp4_a3 --seed 0

# remove explicit group_id
python run_adp4.py --stage adp4_b0 --seed 0

# discover comparable TP edges from S + A
python run_adp4.py --stage adp4_b1 --seed 0

# full automatic comparison graph
python run_adp4.py --stage adp4_b2 --seed 0
```

---

# 30. 多 seed 规则

先只跑：

```text
seed 0
```

只有：

```text
A1 有明显正结果
```

才继续 A2。

只有：

```text
A2 PASS
```

才继续 A3/B。

中等规模：

```text
seeds = 0,1,2
```

至少：

```text
2/3
```

通过后再做：

```text
seeds = 0..9
```

Strong pass：

\[
RecoveryRate\ge 8/10.
\]

---

# 31. Adp4 最终要回答的问题

Adp4 不是要证明“分组比 EM 好”。

它要回答：

\[
\boxed{
\text{高层 }Z_D\text{ 的 identity 是否可以从一批局部可比 TP 中先形成局部作用单元，}
}
\]

再通过：

\[
\boxed{
\text{跨局部邻域的功能对齐}
}
\]

形成全局稳定机制。

如果 A 成功、B 也成功，则支持未来完整架构：

\[
\boxed{
\text{经验流}
\rightarrow
\text{自动发现可比较 TP}
\rightarrow
\text{local mechanism units}
\rightarrow
Z_D
\rightarrow
R_D
\rightarrow
W_D.
}
\]

其中 action / intervention 的角色是：

\[
\boxed{
\text{帮助发现哪些 TP 构成有信息的对照实验，}
}
\]

而不是：

\[
\boxed{
A=Z_D.
}
\]

---

# 32. 一句话执行原则

\[
\boxed{
\text{先用 oracle comparability 证明“local unit → Z_D”算法成立，}
\]
\[
\boxed{
\text{再去掉 oracle，让 learner 从 }S\text{ 和 }A\text{ 自己找到应该比较的 TP。}
}
\]

只有这两步都成立，组内匹配才算真正与未来整体架构对齐。
