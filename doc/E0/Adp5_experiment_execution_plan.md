# Adp5 实验执行方案：以“局部作用族”作为 \(Z_D\) 身份形成的低层证据单位

> **实验目标**
>
> Adp5 用来验证当前最新假设：
>
> \[
> \boxed{
> \text{高层 }Z_D\text{ 不应从单个 TP、单个 fragment 或单点 local unit 直接形成，}
> }
> \]
>
> \[
> \boxed{
> \text{而应先从同一局部作用在多个 realization 下形成的 response family 中获得身份。}
> }
> \]
>
> Adp5 同时主动消除 Adp4 中“每个 base 固定出现 3 个 local units”对 \(M_{\rm real}=3\) 的隐性泄露。
>
> 第一阶段继续使用：
>
> \[
> \boxed{R_D=0}
> \]
>
> 因此只研究 \(Z_D\) identity formation，不研究机制间关系。

---

# 0. Adp5 的实验依据

已有实验得到以下链条：

## E0 / Adp1

从完整 TP 直接学习：

\[
(S,p)
\rightarrow
(m,v)
\rightarrow
\sum_jm_j\mathcal M_j(S,v_j)
\]

会形成 distributed code。

结论：

\[
\boxed{
\text{完整 transformation reconstruction 不足以定义 mechanism identity。}
}
\]

## Adp2

局部 assignment margin 与跨 state context reuse 都不能可靠地产生 GT-like identity。

结论：

\[
\boxed{
\text{模型内部“确定”或“局部可复用”不等于高层 mechanism identity。}
}
\]

## Adp3

将完整 TP 改为 clean single-change fragment 后，至少一个 GT-like mechanism 可以形成。

但同一真实 mechanism 会被多个 candidate 分摊。

结论：

\[
\boxed{
\text{clean fragment 明显降低难度，但单点 fragment 仍不足。}
}
\]

## Adp4

将 4 条重复 fragment 合成 local unit，并在 base 内做联合匹配后：

- reconstruction 更好；
- assignment 更稳定；
- 但 \(v\)-sign split 达到 1.0；
- 同一真实 \(Z_j\) 的正、负 realization 被稳定分给不同 candidate。

因此：

\[
\boxed{
\text{单点 local unit 仍容易把同一高层 }Z_D
\text{ 按 realization 区域切碎。}
}
\]

Adp5 专门验证：

\[
\boxed{
\text{如果一个学习单位自身就同时覆盖多个 realization，
是否能够阻止这种 fragmentation？}
}
\]

---

# 1. 当前对 \(Z_D\) 的工作定义

在 Adp5 中：

\[
\boxed{
Z_D^j
\text{ 是在领域内跨 TP、跨状态、跨时间可复用的一类高层 transformation rule。}
}
\]

具体 realization：

\[
v_i^j
\]

可以变化。

因此：

\[
\boxed{
v<0,\quad v>0,\quad |v|\text{ 不同}
}
\]

不能仅因为表现不同就自动形成不同 \(Z_D\)。

Adp5 要验证的核心性质是：

\[
\boxed{
\text{同一高层 mechanism 是否能统一解释同一局部作用在多个 realization 下的完整响应族。}
}
\]

---

# 2. Adp5 的核心改变

Adp4 的最小学习单位是：

\[
u_g=(S_g,d_g).
\]

Adp5 改为：

\[
\boxed{
U_{gq}
=
\left\{
(S_g,A_{gq,r},d_{gq,r})
\right\}_{r=1}^{R}
}
\]

其中 \(U_{gq}\) 是：

> 在同一局部状态 \(S_g\) 下，同一局部干预方向 \(q\) 以多个 realization 强度出现时产生的一整族 transformation response。

因此学习单位从：

```text
one transformation point
```

升级为：

```text
one local response family
```

---

# 3. 必须消除的 \(M_{\rm real}\) 泄露

Adp4 中每个 base 总能形成恰好 3 个 local units。

这在当前 synthetic world 中会泄露：

\[
\boxed{
\text{“局部存在 3 个不同作用”}
}
\]

的信息。

Adp5 中禁止这个条件。

## 3.1 每个 base 只观察随机子集

真实 world 仍有：

\[
M_{\rm real}=3
\]

但 learner 不知道。

对每个 base 随机选择：

\[
K_g\in\{1,2\}.
\]

第一版：

```text
P(K_g=1) = 0.5
P(K_g=2) = 0.5
```

然后从三个 GT mechanisms 中均匀随机选择 \(K_g\) 个，仅为它们生成 response family。

**任何 base 都不同时展示全部 3 个机制。**

因此 learner 最多只能从一个 base 看到：

```text
1 或 2 个局部作用族
```

不能从单个 base 推断：

\[
M_{\rm real}=3.
\]

## 3.2 全局 coverage 检查

generator 必须确保 train split 中三个 GT mechanism family 的出现次数近似平衡：

```text
max count / min count < 1.15
```

这个 balancing 只在 generator 中实现，不暴露给 learner。

---

# 4. 保持不变的 E0 world

```text
WORLD_SEED = 20260901
DATASET_SEED = 20260902
M_real = 3              # evaluator only
M_max = 5               # learner only knows upper bound
state slots = 3
slot dim = 2
TP = identity
R_D = 0
mechanism architecture = existing SetMechanismBank
```

禁止修改：

```text
GT mechanism functions
A_jk / b_jk / c_jk
state distribution
original active-v support
heldout 101 definition
```

---

# 5. Response family 的 realization 采样

不改变原 active realization 支持：

\[
v\in[-1,-0.3]\cup[0.3,1].
\]

每个 local response family 使用一组固定的 **learner 不可见 GT amplitudes**：

```text
TRAIN ACTIVE VALUES:
[-1.00, -0.70, -0.40, +0.40, +0.70, +1.00]

HELDOUT FAMILY VALUES:
[-0.85, -0.55, -0.30, +0.30, +0.55, +0.85]
```

此外加入：

```text
baseline = mechanism absent
```

即：

\[
m_j=0.
\]

baseline 不是新的 \(v\) 数据，而是原 E0 中 mechanism 不参与时已有的零作用条件。

---

# 6. 每个 response family 如何生成

对于 base \(g\)：

先采样：

\[
S_g.
\]

对隐藏选择的 mechanism \(j\)：

固定其它 mechanism 为 absent。

对每个 active realization：

\[
v_r
\]

生成：

\[
\Delta S_{g,j,r}
=
\mathcal M_j^*(S_g,v_r).
\]

baseline：

\[
\Delta S_{g,0}=0.
\]

定义 response：

\[
\boxed{
d_{g,j,r}
=
\Delta S_{g,j,r}
-
\Delta S_{g,0}.
}
\]

由于：

\[
R_D=0,
\]

这里就是该 mechanism 在：

\[
(S_g,v_r)
\]

下的直接 effect。

---

# 7. learner 可以看到什么

Adp5-A 阶段允许 learner 看到：

```text
base-local family_id
S_pre
action / intervention observation A
DeltaS / TP-derived effect d
within-family member relation
```

禁止 learner 看到：

```text
GT mechanism index j
GT m
GT v
GT effect label
global correspondence between family_id values
M_real
```

特别注意：

\[
\boxed{
family\_id
}
\]

只能在一个 base 内有效。

例如：

```text
base 17: family 0
base 18: family 0
```

不表示它们是同一个 \(Z_D\)。

每个 base 的 local family labels 必须随机重新排列。

---

# 8. Adp5 总体 stage

```text
Adp5-0   数据与泄露审计
   ↓
Adp5-A0  response family 本身是否覆盖完整 realization 变化
   ↓
Adp5-A1  family-level mechanism discovery，Mmax=5，不做数量合并
   ↓
Adp5-A2  跨 base heldout family reuse 诊断
   ↓
Adp5-A3  functional merge，自动决定全局机制数量
   ↓
Adp5-B1  用学出的 Z_D 回到完整 TP 推断 m,v
   ↓
Adp5-B2  小幅联合 refinement
   ↓
Adp5-C   去掉 oracle family grouping，自动发现 response family
   ↓
Adp5-F   多 seed
```

任何一步 FAIL：

```text
STOP
```

先解释，不自动进入下一步。

---

# 9. Adp5-0：数据与泄露审计

不训练。

必须输出：

```text
base_count
K_g histogram
GT family counts evaluator-only
families per base
members per family
global GT coverage
```

检查：

1. 每个 base：
   \[
   K_g\in\{1,2\};
   \]
2. 不存在任何：
   \[
   K_g=3;
   \]
3. 三个 GT family 全局近似平衡；
4. local family id 在不同 base 之间随机重排；
5. learner-facing 文件中不存在：
   ```text
   GT j
   GT v
   GT m
   M_real
   ```

### PASS

所有检查必须 PASS。

---

# 10. Adp5-A0：response family 完整性

目的：

\[
\boxed{
\text{验证同一个 local family 确实同时包含正、负和不同幅度 realization。}
}
\]

每个 family：

```text
6 train active points
6 heldout active points
1 baseline
```

evaluator 检查：

```text
contains negative side
contains positive side
covers low/mid/high |v|
```

同时检查同一个 family 的 GT mechanism index 恒定。

### PASS

```text
family purity = 1.0
100% families cover both signs
100% families cover >= 3 magnitude regions
```

---

# 11. Adp5-A1：family-level mechanism discovery

这是 Adp5 核心实验。

维护：

\[
M_{\max}=5
\]

个 global candidate mechanisms：

\[
\mathcal M_1,\ldots,\mathcal M_5.
\]

---

# 12. 一个 response family 只能属于一个 candidate

对：

\[
U_{gq}
\]

只允许：

\[
c_{gq}\in\{1,\ldots,5\}.
\]

整个 family：

```text
negative realizations
positive realizations
small magnitude
large magnitude
```

必须共享：

\[
\boxed{c_{gq}}.
\]

因此禁止：

```text
v<0 → C1
v>0 → C4
```

这种 Adp4 的 sign fragmentation 在同一个 family 内发生。

---

# 13. family 内每个 response point 仍有自己的 realization latent

虽然 candidate assignment 共享，但每个成员仍允许：

\[
v_{gq,r}
\]

不同。

candidate \(j\) 对 family \(U\) 的解释代价：

\[
\boxed{
E(U,j)
=
\min_{\{v_r\}}
\frac1R
\sum_{r=1}^{R}
D_S
\left(
d_r,
\mathcal M_j(S_U,v_r)
\right).
}
\]

第一版继续：

```text
v = 1.5 tanh(z)
inner Adam lr = 0.05
inner steps = 20
restarts = 2
```

**不要强制 learner latent \(v_r\) 等于 generator 的 GT realization。**

---

# 14. family-level assignment

冻结 candidates。

对每个 family：

\[
\boxed{
c_U
=
\arg\min_j E(U,j).
}
\]

Adp5-A1 **不使用 base 内 injective constraint**。

原因：

1. 每个 base 只观察 1 或 2 个 family；
2. 不希望再把“这个 base 有几个局部 unit”写成全局 \(Z_D\) 数量先验；
3. 是否属于不同 global mechanism 应由跨 base 的功能规律决定，而不是 base 内固定编号。

---

# 15. M-step

对 candidate \(j\)：

\[
\mathcal D_j
=
\{U:c_U=j\}.
\]

使用 family 中所有 train-realization points 更新：

\[
\mathcal M_j.
\]

配置：

```text
optimizer = AdamW
lr = 3e-4
weight_decay = 1e-5
batch = 32 families
rounds = 40
epochs_per_round = 4
```

如果显存允许，可提升到：

```text
batch = 64 families
```

但科学设置保持不变。

---

# 16. 关键：family 内 heldout realization 不参与 M-step

每个 family 的：

```text
TRAIN ACTIVE VALUES
```

参与 candidate fitting。

而：

```text
HELDOUT FAMILY VALUES
```

只用于评价：

> 一个 candidate 是否学到了“整条局部作用规律”，而不是只记住给定 realization 点。

定义：

\[
E_{\rm interp}(U,j)
=
\frac1{R_h}
\sum_{r\in heldout}
D_S
\left(
d_r,
\mathcal M_j(S_U,\tilde v_r)
\right),
\]

其中只重新优化：

\[
\tilde v_r,
\]

禁止更新：

\[
\mathcal M_j.
\]

---

# 17. A1 必须记录

每轮：

```text
round

train family MSE
heldout-realization family NRMSE

candidate family usage[5]
candidate family count[5]

family assignment change rate
exact family assignment agreement

functional R2 matrix [5x3]
matched functional R2

family Hungarian accuracy
family contingency matrix

candidate state coverage
candidate effect-norm coverage
```

evaluator-only：

```text
GT-family purity per candidate
v-sign split
v-magnitude split
candidate GT-family confusion
```

---

# 18. Adp5 对 sign split 的定义

虽然一个 family 内已禁止 split，但仍可能出现：

```text
不同 base 的同一 GT mechanism：
负侧更强的 family → C1
正侧更强的 family → C4
```

由于每个 family 本身已经同时覆盖正负，这种情况理论上应显著减少。

仍然计算：

\[
Split_k
=
\frac12
\sum_j
\left|
P(C_j\mid Z_k,\text{negative-dominant evidence})
-
P(C_j\mid Z_k,\text{positive-dominant evidence})
\right|.
\]

并同时报告 family-level GT fragmentation：

\[
Frag_k
=
1-\max_jP(C_j\mid Z_k).
\]

---

# 19. A1 PASS

development seed 0：

```text
family Hungarian accuracy > 0.80
三个 GT 各自 best functional R² > 0.70
至少两个 matched functional R² > 0.85
last-3 family assignment change < 0.15
mean GT fragmentation < 0.30
```

重点：

\[
\boxed{
\text{不要求 alive candidate count = 3。}
}
\]

A1 只解决 identity formation。

---

# 20. A1 与 Adp4 的直接比较

必须生成：

| 指标 | Adp4-A1 | Adp5-A1 |
|---|---:|---:|
| validation NRMSE |  |  |
| Hungarian accuracy |  |  |
| matched functional R² |  |  |
| assignment change |  |  |
| sign split |  |  |
| GT fragmentation |  |  |

核心判据不是 reconstruction，而是：

\[
\boxed{
\text{是否真正消除了 realization-driven identity fragmentation。}
}
\]

---

# 21. Negative control NC-1：打散 family

保持所有 response point 不变。

但将一个 local response family 的不同 realization 随机拆成独立 point assignment。

等价于退回：

```text
single-point local unit
```

预期：

```text
sign fragmentation 上升
functional recovery 下降
```

若 Adp5 主实验与 NC-1 无差异，则 response-family 假设没有得到支持。

---

# 22. Negative control NC-2：只给单侧 realization

训练 family 只包含：

```text
negative values
```

或：

```text
positive values
```

随机各一半 family。

预期重新出现：

\[
Z_j^-,\quad Z_j^+
\]

分裂。

这是验证：

\[
\boxed{
\text{“同时观察 realization 的多个区域”是否是 identity 形成的关键证据。}
}
\]

---

# 23. Adp5-A2：跨 base reuse 诊断

只有 A1 有明显正结果才执行。

目标：

\[
\boxed{
\text{验证一个 candidate 是否真的跨不同状态解释完整 response family。}
}
\]

不能使用“candidate 在自己训练过的 family 上误差低”作为证据。

---

# 24. Cross-base split

按 base 划分：

```text
fit bases
evidence bases
```

第一版：

```text
50 / 50
```

对 candidate \(j\)：

1. 仅在 fit bases 中属于 \(j\) 的 families 更新临时 mechanism copy；
2. 冻结 copy；
3. 到 evidence bases 上重新优化每个 response point 的 realization latent；
4. 计算完整 family heldout error。

定义：

\[
E_j^{cross-base}.
\]

同时反向：

```text
evidence → fit
```

取平均。

---

# 25. A2 PASS

对于 functional \(R^2\) 高的 candidate：

```text
cross-base heldout NRMSE 显著低于 random-family / wrong-candidate baseline
```

并要求 learner-visible reuse score 与 evaluator functional \(R^2\) 呈正确趋势。

如果再次出现：

```text
reuse error 下降但 functional R² 变差
```

则 STOP。

说明 family-level reuse 仍不能作为 global identity evidence。

---

# 26. Adp5-A3：全局 mechanism 数量选择

只有：

```text
A1 identity positive
+
A2 cross-base reuse valid
```

后执行。

---

# 27. 全局存在状态

维护：

\[
g_j\in\{0,1\}.
\]

```text
1 = alive
0 = dormant
```

初始：

\[
g_j=1,\quad j=1,\ldots,5.
\]

---

# 28. functional merge

对 candidates \(j,k\)：

取二者全部 response families：

\[
\mathcal D_j\cup\mathcal D_k.
\]

训练一个临时 merged mechanism：

\[
\mathcal M_{j\oplus k}.
\]

必须按 **base** 做 heldout validation。

比较：

\[
L_{\rm sep}^{heldout}
\]

与：

\[
L_{\rm merge}^{heldout}.
\]

加入全局 mechanism cost：

\[
\lambda_G.
\]

接受 merge 当：

\[
\boxed{
L_{\rm merge}^{heldout}
-
L_{\rm sep}^{heldout}
<
\lambda_G.
}
\]

---

# 29. \(\lambda_G\) 选择

development seed 0：

```text
lambda_G ∈
{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
```

不能根据：

```text
最终是否恰好 3 个
```

选择。

规则：

```text
在 heldout family NRMSE 距最优 <= 2% 的配置中，
选择 alive mechanism 数最少者。
```

---

# 30. A3 PASS

```text
alive candidate count = 3
三个 matched functional R² > 0.90
family Hungarian accuracy > 0.90
false merge = 0
cross-base reuse 保持
```

---

# 31. Adp5-B1：回到完整 TP

只有 A3 PASS 后执行。

冻结学出的：

\[
Z_D=\{\mathcal M_1,\mathcal M_2,\mathcal M_3\}.
\]

对原 E0 full TP：

\[
(S_i,p_i,\Delta S_i)
\]

推断：

\[
m_i,v_i.
\]

枚举：

\[
2^3=8
\]

个 support。

目标：

\[
\boxed{
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
}
\]

---

# 32. B1 PASS

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
\text{response-family 形成的高层 }Z_D
\text{ 是否能重新组合解释完整 TP。}
}
\]

---

# 33. Adp5-B2：小幅 joint refinement

只有 B1 PASS 后执行。

使用：

\[
L_{\rm effect}
+
\lambda_PL_{\rm participation}.
\]

mechanism learning rate：

```text
= discovery stage lr × 0.1
```

监控：

```text
functional R² drift
participation F1
IID / 101 NRMSE
```

### PASS

```text
min functional R² 下降 < 0.05
prediction 不恶化
participation 不恶化
```

---

# 34. TP reconstruction bridge

B2 稳定后再补：

\[
L_{\rm TP-rec}.
\]

比较：

\[
L_{\rm effect}
+
L_{\rm participation}
\]

与：

\[
L_{\rm effect}
+
L_{\rm participation}
+
L_{\rm TP-rec}.
\]

目的：

\[
\boxed{
\text{验证完整母模型中的 TP reconstruction 是否保持信息而不破坏已经形成的 }Z_D.
}
\]

不允许用 TP-rec loss 本身下降作为成功判据。

---

# 35. Adp5-C：去掉 oracle response-family grouping

只有 A/B 成功后执行。

这是与未来整体架构对齐的关键 bridge。

learner 不再看到：

```text
family_id
within-family relation
```

只看到：

\[
\boxed{
(S^{pre},A,S^{post})
}
\]

或：

\[
(S^{pre},A,p,\Delta S).
\]

---

# 36. 自动发现 local response family

在相近状态 neighborhood 内：

1. 收集多个 TP；
2. 计算 action difference；
3. 找重复出现的局部 intervention directions；
4. 沿同一 direction 按强度排序；
5. 对应的 TP responses 形成 candidate response family；
6. 再输入 Adp5-A1 的 family-level mechanism learner。

---

# 37. action 的角色

必须保持：

\[
\boxed{
A\neq Z_D.
}
\]

action 只用于：

\[
\boxed{
\text{判断哪些 TP 是同一个局部 intervention family 的不同 realization。}
}
\]

禁止直接把 \(A\) 输入：

\[
\mathcal M_j(S,v)
\]

作为 shortcut。

---

# 38. C 阶段 PASS

自动 family discovery：

```text
family pair precision > 0.90
family pair recall > 0.85
```

最终：

```text
matched functional R² each > 0.85
family accuracy > 0.85
full TP participation F1 > 0.85
101 NRMSE < 0.12
```

---

# 39. 数据规模

## development

```text
train bases = 1024
val bases = 256
test bases = 512

K_g ∈ {1,2}
average families/base ≈ 1.5

train realization points/family = 6
heldout points/family = 6
```

约：

```text
train families ≈ 1536
train response points ≈ 9216
```

规模低于 Adp3 fragment count，可直接完整运行。

---

# 40. 中等规模

只有 development seed 0 有明确正结果后：

```text
train bases = 4096
val bases = 1024
test bases = 2048

optimization seeds = 0,1,2
```

至少：

```text
2/3
```

通过才继续。

---

# 41. 正式多 seed

```text
optimization seeds = 0..9
```

Strong pass：

\[
\boxed{
RecoveryRate\ge8/10.
}
\]

---

# 42. 输出目录

```text
outputs/
└── e0_adp5/
    ├── adp5_0/
    ├── a0_response_families/
    ├── a1_family_discovery/
    │   ├── checkpoints/
    │   ├── assignments/
    │   ├── round_metrics.jsonl
    │   └── final_metrics.json
    ├── a2_cross_base/
    ├── a3_merge/
    ├── b1_full_tp/
    ├── b2_refinement/
    ├── c_auto_family/
    └── negatives/
```

---

# 43. 每轮必须保存

```text
round

train_family_mse
heldout_realization_nrmse

candidate_family_usage[5]
candidate_family_count[5]

family_assignment_change_rate
family_exact_agreement

family_hungarian_accuracy
family_contingency

functional_r2_matrix
matched_functional_r2

candidate_state_coverage

GT_fragmentation             # evaluator only
sign_split                    # evaluator only
magnitude_split               # evaluator only

runtime
```

---

# 44. 推荐命令模板

```bash
python run_adp5.py --stage adp5_0 --seed 0
python run_adp5.py --stage adp5_a0 --seed 0
python run_adp5.py --stage adp5_a1 --seed 0
python run_adp5.py --stage adp5_a2 --seed 0
python run_adp5.py --stage adp5_a3 --seed 0
python run_adp5.py --stage adp5_b1 --seed 0
python run_adp5.py --stage adp5_b2 --seed 0
python run_adp5.py --stage adp5_c --seed 0
```

---

# 45. 关键结果表

最终必须生成：

| 方法 | 最小学习单位 | Mreal 泄露 | functional recovery | sign split | count recovery | full TP |
|---|---|---:|---:|---:|---:|---:|
| Adp3 | fragment | 有局部结构 | 1 个较好 | 高 | FAIL | N/A |
| Adp4 | point local unit | 较强 | FAIL | 1.0 | N/A | N/A |
| Adp5-A1 | response family | 显著降低 | ? | ? | 不要求 | N/A |
| Adp5-A3 | response family | 显著降低 | ? | ? | ? | N/A |
| Adp5-B1 | learned \(Z_D\) | 无直接 Mreal | ? | ? | ? | ? |
| Adp5-C | auto family | 无 oracle family | ? | ? | ? | ? |

---

# 46. Adp5 的失败分支

## A1 FAIL：sign split 仍高

说明：

\[
\boxed{
\text{即使一个 local family 同时覆盖多个 realization，
当前 representation 仍无法把它们视为同一高层机制。}
}
\]

下一步应直接检查：

\[
\mathcal M_j(S,v)
\]

本身对 realization family 的参数化是否合适。

不要进入 merge。

---

## A1 identity 好，但有 4/5 个 candidate

说明：

\[
\boxed{
\text{高层 family identity 已形成，global count 尚未解决。}
}
\]

进入 A2/A3。

---

## A2 reuse 再次与 functional identity 反向

说明：

\[
\boxed{
\text{即便 response family 完整，单纯跨状态 reuse 仍不是可靠 identity 证据。}
}
\]

STOP。

---

## A3 count=3 但 functional recovery 差

不算成功。

说明全局 cost 只把错误模型压缩成 3 个。

---

## B1 FAIL

说明：

\[
\boxed{
\text{response-family identity 可以形成，
但多个高层机制重新组合成完整 TP 仍存在 composition 问题。}
}
\]

此时才重新研究 participation。

---

## C FAIL

如果 oracle family 成功、自动 family discovery 失败：

\[
\boxed{
\text{核心 }Z_D\text{ learner 可行，瓶颈转移到 exploration / comparison discovery。}
}
\]

这与未来主动探索模块直接对接。

---

# 47. Adp5 最终要回答的科学问题

Adp5 不是在验证一种新的聚类技巧。

它在验证：

\[
\boxed{
\text{一个高层 }Z_D
\text{ 的低层身份依据，是否应该是一整族随 realization 变化的局部 response，}
}
\]

而不是：

\[
\boxed{
\text{单个 transformation point。}
}
\]

如果成功，则学习顺序更新为：

\[
\boxed{
\text{TP experience}
\rightarrow
\text{local comparable interactions}
\rightarrow
\text{local response families}
\rightarrow
\text{cross-state/global alignment}
\rightarrow
Z_D
}
\]

随后：

\[
\boxed{
Z_D
\rightarrow
\text{full TP participation}
\rightarrow
R_D
\rightarrow
W_D.
}
\]

---

# 48. 一句话执行标准

\[
\boxed{
\text{先证明“一个同时覆盖多 realization 的 response family 能形成稳定 }Z_D\text{ 身份”，}
}
\]

\[
\boxed{
\text{再解决全局数量、完整 TP 组合和自动 family discovery。}
}
\]

Adp5 的首要成功标志不是 reconstruction 更低，而是：

\[
\boxed{
\text{Adp4 中由 realization 导致的 }Z_D\text{ fragmentation 被真正消除。}
}
