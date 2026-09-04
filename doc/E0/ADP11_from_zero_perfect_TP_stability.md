# ADP11 实验执行方案：Perfect-TP 条件下的 From-Zero 稳定性与正交 TP 对照

## 0. 实验目标

ADP10 已证明：在高质量 response-family evidence 已给定时，ADP6→ADP9→B1 后半链路具有较高优化稳定性。ADP11 进一步检验：

\[
\boxed{
R_D=0,\ \text{TP 完美时，整套 }Z_D\text{ discovery 是否能从随机初始化稳定恢复？}
}
\]

同时比较两种信息等价的 TP：

### Identity TP
\[
p_i^{id}=\operatorname{vec}(\Delta S_i)
\]

### Orthogonal TP
\[
p_i^{orth}=B\operatorname{vec}(\Delta S_i),\qquad B^\top B=I
\]

其中 \(B\) 是一次生成后固定的随机正交矩阵。

本实验回答两个问题：

1. **From-zero stability**：是否能从随机 \(M_{\max}=5\) mechanism bank 自主形成稳定 \(Z_D\)；
2. **TP representation invariance**：机制发现是否依赖 TP 坐标轴恰好与 \(\Delta S\) 对齐。

---

# 1. 结论边界

若 ADP11 strong pass，只能支持：

\[
\boxed{
\text{在 }R_D=0\text{、TP 无信息损失的理想条件下，当前 }Z_D
\text{ discovery procedure 稳定。}
}
\]

它**不能**证明：

```text
visual state encoder 已稳定
TP segmentation 已稳定
learned TP 已稳定
R_D discovery 已稳定
完整 embodied system 已稳定
```

---

# 2. 世界与数据

沿用 E0-A：

\[
S=\{s^1,s^2,s^3\},\qquad s^k\in\mathbb R^2,\qquad s^k\sim U(-1,1)^2
\]

真实机制数：

\[
M^*=3
\]

learner：

\[
M_{\max}=5
\]

incidence：

\[
C=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&0&1
\end{bmatrix}
\]

realization：

\[
v_j\sim U([-1,-0.3]\cup[0.3,1])
\]

effect：

\[
e_j^k=
m_jC_{jk}
\left[
v_jA_{jk}s^k+v_jb_{jk}+0.25v_j^2c_{jk}
\right]
\]

\[
\Delta S=e_1+e_2+e_3,\qquad R_D=0
\]

固定：

```text
WORLD_SEED
DATASET_SEED
train/val/test split
base states
participation sampling
response-family construction
heldout 101
family/base split
```

只改变：

```text
optimization_seed
parameter initialization
minibatch order
optimizer stochasticity
TP condition
```

---

# 3. Orthogonal TP 构造

生成：

\[
G_{ab}\sim\mathcal N(0,1)
\]

做 QR：

\[
G=QR
\]

并采用 deterministic sign convention 得到 \(B=Q\)。

要求：

```text
B 跨 optimization seeds 固定
保存 B.npy 和 hash
learner 只看到 p，不得到 B 的语义解释
禁止显式 inverse-B 恢复 ΔS
```

Identity 与 Orthogonal 条件除 TP 坐标外使用完全相同的数据。

---

# 4. 实验矩阵

## Pilot

```text
Identity TP:   seeds 0,1,2
Orthogonal TP: seeds 0,1,2
```

两个条件都达到：

\[
\boxed{\ge2/3}
\]

才进入正式 10-seed。

## Strong test

```text
Identity TP:   seeds 0..9
Orthogonal TP: seeds 0..9
```

每个条件 strong pass：

\[
\boxed{\ge8/10}
\]

---

# 5. “From Zero”的严格定义

每个 run 必须：

```text
随机初始化 Mmax=5 mechanism bank
随机初始化 family-to-candidate assignment
随机初始化 candidate coordinate parameters

不加载 ADP5 mechanism checkpoint
不加载 ADP6 checkpoint
不加载 ADP9 graph
不硬编码 duplicate pair
```

允许使用：

```text
perfect TP
base-local family grouping
learner-visible action/intervention-derived alpha
```

禁止 learner 使用：

```text
GT mechanism label
GT v
GT m
GT duplicate pair
M_real=3
```

\(M_{\rm real}=3\) 仅 evaluator 最终检查。

---

# 6. 总体 pipeline

```text
Z0  初始化与数据审计
 ↓
Z1  response-family 构建
 ↓
Z2  from-zero family-level identity formation
 ↓
Z3  shared realization coordinate
 ↓
Z4  joint family assignment refinement
 ↓
Z5  functional duplicate graph
 ↓
Z6  gradual duplicate pruning
 ↓
Z7  final family stability
 ↓
Z8  full-TP B1 composition
 ↓
FINAL
```

---

# 7. Z0：审计

保存：

```text
condition
optimization_seed
initial model hash
initial assignment hash
dataset hash
family hash
B hash（orthogonal）
config hash
git commit
```

要求：

```text
同条件不同 seed 的数据 hash 完全相同
不同 seed 的 initial model hash 不同
identity / orthogonal 除 p 以外的数据完全一致
```

---

# 8. Z1：Response Family

沿用 ADP5 的防泄漏设计：

```text
每个 base 只出现 K_g ∈ {1,2}
不让单个 base 暴露全部 3 个真实机制
family_id 仅在 base 内有效并随机置换
```

推荐 realization grid：

```text
train:
[-1.0,-0.7,-0.4,0.4,0.7,1.0]

heldout:
[-0.85,-0.55,-0.3,0.3,0.55,0.85]
```

每个 family 含 absent baseline。

learner-visible：

```text
S
p
A / intervention descriptor
local family_id
response d
```

---

# 9. Z2：From-Zero Family Identity Formation

目标：

\[
\boxed{
\text{从 base-local response families 中形成跨 base 可复用的 global candidates}
}
\]

whole-family assignment：

\[
c_U\in\{1,\dots,M_{\max}\}
\]

同一 local family 的所有 realization points 必须共享 \(c_U\)。

禁止：

```text
per-point mechanism assignment
family 内不同点分到不同 candidate
```

主实验初始化固定为：

\[
\boxed{\text{random balanced family assignment}}
\]

不要根据 seed 选择更有利初始化。

response-feature clustering 只能作为后续 diagnostic，不属于正式主实验。

---

# 10. Z3：Shared Realization Coordinate

从 learner-visible intervention geometry 得到：

\[
\delta A_r=A_r-A_0
\]

每个 local family 做 PC1：

\[
q_U
\]

\[
\alpha_{U,r}=q_U^\top\delta A_r
\]

归一化得到：

\[
\tilde\alpha_{U,r}
\]

candidate-level coordinate：

\[
\boxed{
v_{U,r}=a_{c_U}\tilde\alpha_{U,r}+b_{c_U}
}
\]

禁止：

```text
per-point free v
GT amplitude
GT sign
family-specific arbitrary coordinate
```

---

# 11. Z2-Z4 交替优化

每轮：

```text
1. 固定 assignment
2. 更新 mechanism function + candidate-level a,b
3. 固定 mechanism/coordinate
4. 计算 whole-family cost
5. 重新 assignment
6. 记录 telemetry
```

family cost：

\[
E(U,j)=
\frac1R\sum_r
D_S\left(
d_{U,r},
\mathcal M_j(S_U,a_j\tilde\alpha_{U,r}+b_j)
\right)
\]

assignment：

\[
c_U=\arg\min_jE(U,j)
\]

---

# 12. Z4 Gate：允许 Duplicate，不允许 Mixing

ADP6/10 已表明 duplicate symmetry 是允许的，因此正式 gate **不要求**：

```text
立即 M=3
fragmentation=0
raw assignment 完全稳定
```

要求：

```text
每个 GT mechanism 至少存在一个高质量 functional candidate
purity-high candidates 不发生 mechanism mixing
realization coordinate 稳定
heldout response 足够好
```

evaluator-only PASS：

```text
每个 GT mechanism 的 best functional R² > 0.95
所有主要 purity-high candidate functional R² > 0.90
min affine-aligned realization R² > 0.95
median order accuracy > 0.95
heldout-realization NRMSE < 0.05
无明显 mixed candidate
```

GT 只用于评价，不参与优化。

---

# 13. Z5：Functional Duplicate Graph

对 alive candidates 的所有 pair 计算：

```text
cross-state prediction R²
cross-realization prediction R²
cross-base prediction R²
direct-replacement max ΔNRMSE
```

正式 valid edge：

```text
cross-state R² > 0.995
cross-realization R² > 0.995
cross-base R² > 0.995
direct-replacement max ΔNRMSE < 0.01
```

ADP11 不修改这个 gate。若要做 uncertainty-aware gate，另开下一实验。

---

# 14. Z6：Gradual Duplicate Pruning

沿用 ADP9：

```text
build graph
→ choose highest-confidence valid edge
→ learner-visible survivor selection
→ hard prune one candidate
→ freeze all mechanisms
→ full family reassignment
→ structured validation
→ commit / rollback
→ rebuild graph
```

停止条件：

\[
\boxed{\text{no valid duplicate edge}}
\]

禁止：

```text
alive_count == 3 时强制停止
```

---

# 15. 单步 Prune Gate

每次删除必须满足：

```text
max structured NRMSE increase < 0.01
每个 GT mechanism 的 best functional R² > 0.95
family Hungarian accuracy drop < 0.03
fragmentation increase < 0.03
repeated reassignment change < 0.10
```

若 FAIL：

```text
rollback
blacklist pair at current round
继续测试下一条 valid edge
```

---

# 16. Z7：最终 Family-Level Gate

pruning 自动停止后记录：

```text
M_discovered
stop reason
family Hungarian accuracy
mean fragmentation
assignment change
matched functional R²
IID NRMSE
cross-state NRMSE
cross-realization NRMSE
cross-base NRMSE
```

PASS：

```text
stop reason = no valid duplicate edge
family Hungarian accuracy > 0.90
mean fragmentation < 0.10
assignment change < 0.10
all matched functional R² > 0.95
IID NRMSE < 0.03
cross-state NRMSE < 0.05
cross-realization NRMSE < 0.05
cross-base NRMSE < 0.05
```

最后 evaluator 才检查：

```text
M_discovered == 3 ?
```

---

# 17. Z8：Full-TP B1

使用该 run 自己 surviving mechanisms。

枚举：

\[
2^{M_{\rm discovered}}
\]

个 support。

对 active mechanism 优化：

\[
\alpha_i^j
\]

并使用：

\[
v_i^j=a_j\alpha_i^j+b_j
\]

目标：

\[
\min_{m,\alpha}
D_S\left(
\Delta S_i,
\sum_jm_i^j
\mathcal M_j(S_i,a_j\alpha_i^j+b_j)
\right)
+
\lambda_P\|m_i\|_0
\]

固定 solver：

```text
inner steps = 60
restarts = 3
chunk size = 1024
same lambda_P as ADP10
```

禁止单独给失败 seed 增加预算。

B1 PASS：

```text
IID NRMSE < 0.05
101 NRMSE < 0.10
participation micro-F1 > 0.90
min active instance-effect R² > 0.90
min functional R² > 0.90
inactive Z2 FPR on 101 < 0.10
```

---

# 18. 单 Run 最终 PASS

一个 condition-seed run 必须同时满足：

```text
Z4 identity/coordinate PASS
Z6 pruning 正常完成
Z7 final family stability PASS
M_discovered = 3           # evaluator-only final check
Z8 B1 PASS
```

---

# 19. Pilot 与 Strong Pass

## 3-seed Pilot

Identity：

\[
\boxed{\ge2/3}
\]

Orthogonal：

\[
\boxed{\ge2/3}
\]

两个条件都满足才扩 10 seeds。

## 10-seed Strong Pass

Identity：

\[
\boxed{\ge8/10}
\]

Orthogonal：

\[
\boxed{\ge8/10}
\]

并记录：

\[
|\hat p_{id}-\hat p_{orth}|
\]

第一版要求：

\[
|\hat p_{id}-\hat p_{orth}|\le0.20
\]

---

# 20. TP Representation Invariance 不只看 Pass Rate

paired-by-seed 比较：

```text
M_discovered
rounds to functional recovery
rounds to pruning convergence
matched functional R²
family accuracy
fragmentation
B1 IID NRMSE
B1 101 NRMSE
participation F1
```

定义：

\[
\Delta metric_s
=
metric_s^{orth}
-
metric_s^{id}
\]

重点看：

```text
effect size
failure-mode shift
recovery-rate gap
```

---

# 21. Negative Controls

## NC-1：Non-invertible TP

\[
p=P\Delta S,\qquad \operatorname{rank}(P)<\dim(\Delta S)
\]

预期 recovery 显著下降。

用于证明 perfect TP 的信息保真性确实重要。

## NC-2：Shuffle Family Grouping

打乱 learner-visible local family grouping。

若性能下降，支持 response-family evidence 的必要性。

## NC-3：Free-v

恢复 per-point free \(v\)。

预期 cross-family realization gauge / functional recovery 下降。

## NC-4：No Pruning

保留全部 5 candidates 直接执行 B1。

验证 duplicate ambiguity 是否影响 global count / participation。

## NC-5：一般可逆但非正交 TP（可选 diagnostic）

\[
p=C\Delta S
\]

其中 \(C\) 可逆但 condition number 较高。

不纳入 ADP11 主结论，仅用于探索 orthogonal invariance 是否可推广到一般可逆重参数化。

---

# 22. 失败分类

## F0：TP-coordinate sensitivity
Identity PASS，Orthogonal 系统性 FAIL。

## F1：Family identity failure
无法形成 purity-high reusable candidates。

## F2：Realization coordinate failure
order / affine alignment 失败。

## F3：Mechanism mixing
candidate 混入多个真实 functional identities。

## F4：Duplicate graph false negative
明显 duplicate 没有建边，最终 \(M_{\rm discovered}>3\)。

## F5：Duplicate graph false positive
不同机制被错误 prune。

## F6：Reassignment instability
pruning 后 family assignment 不稳定。

## F7：Full-TP composition failure
family-level \(Z_D\) 正确，但 B1 失败。

---

# 23. 输出目录

```text
outputs/
└── e0_adp11_from_zero/
    ├── identity_tp/
    │   ├── seed_000/
    │   │   ├── z0_audit/
    │   │   ├── z1_families/
    │   │   ├── z2_identity/
    │   │   ├── z3_coordinate/
    │   │   ├── z4_joint/
    │   │   ├── z5_graph/
    │   │   ├── z6_pruning/
    │   │   ├── z7_final/
    │   │   ├── z8_b1/
    │   │   └── final_verdict.json
    │   └── ...
    ├── orthogonal_tp/
    │   ├── B.npy
    │   ├── seed_000/
    │   └── ...
    ├── aggregate_3seed/
    ├── aggregate_10seed/
    └── negatives/
```

---

# 24. 每个 Run 的 Summary

```json
{
  "condition": "identity_tp",
  "seed": 0,
  "z4_identity_pass": true,
  "z4_coordinate_pass": true,
  "pruning_pass": true,
  "m_discovered": 3,
  "family_accuracy": 1.0,
  "fragmentation": 0.0,
  "b1_pass": true,
  "final_pass": true,
  "failure_mode": null
}
```

---

# 25. 汇总表

| TP | Seed | Z4 Identity | Coord | Final M | Prune | B1 IID | B1 101 | F1 | Final |
|---|---:|---|---|---:|---|---:|---:|---:|---|
| Identity | 0 |  |  |  |  |  |  |  |  |
| Identity | 1 |  |  |  |  |  |  |  |  |
| Identity | 2 |  |  |  |  |  |  |  |  |
| Orthogonal | 0 |  |  |  |  |  |  |  |  |
| Orthogonal | 1 |  |  |  |  |  |  |  |  |
| Orthogonal | 2 |  |  |  |  |  |  |  |  |

---

# 26. 命令模板

```bash
# Pilot
python run_adp11.py --condition identity --seed 0
python run_adp11.py --condition identity --seed 1
python run_adp11.py --condition identity --seed 2

python run_adp11.py --condition orthogonal --seed 0
python run_adp11.py --condition orthogonal --seed 1
python run_adp11.py --condition orthogonal --seed 2

python run_adp11.py --stage aggregate --seeds 0,1,2

# 两边均通过后
python run_adp11.py --stage auto_10seed
```

---

# 27. 必须保存的 Telemetry

每轮至少保存：

```text
family assignment change
candidate usage

family Hungarian accuracy       # evaluator-only
fragmentation                   # evaluator-only
candidate purity                # evaluator-only
functional R² matrix            # evaluator-only

heldout NRMSE
coordinate a,b
affine realization R²
order accuracy

duplicate graph edges
prune decisions
rollback events
```

---

# 28. 禁止的事后修改

正式实验开始后禁止：

```text
orthogonal 较差后只调 orthogonal 学习率
M=4 后放宽 duplicate threshold
失败 seed 单独增加 rounds
失败 seed 单独增加 B1 restarts
使用 GT pair 手动 prune
根据 M_real=3 强制继续删除
```

任何修改必须作为新的实验版本。

---

# 29. 结果解释

### Case A：Identity 与 Orthogonal 都 Strong PASS

\[
\boxed{
\text{perfect TP 条件下，当前 }Z_D\text{ discovery 从零稳定，
且对正交 TP 重参数化近似不变。}
}
\]

### Case B：Identity PASS，Orthogonal FAIL

\[
\boxed{
\text{from-zero discovery 可行，但依赖 TP 坐标结构。}
}
\]

### Case C：两者都在 Z2/Z4 失败

\[
\boxed{
\text{family identity formation 仍不是稳定的 from-zero discovery mechanism。}
}
\]

### Case D：Z4 PASS，Pruning FAIL

\[
\boxed{
\text{functional mechanism formation 稳定，但 global identity compression 不稳定。}
}
\]

### Case E：Pruning PASS，B1 FAIL

\[
\boxed{
\text{global mechanisms 已形成，但 full TP support inference 仍不稳定。}
}
\]

---

# 30. 与理论框架的关系

ADP11 检验的学习链是：

\[
\text{local transformation evidence}
\rightarrow
\text{response families}
\rightarrow
\text{reusable functional hypotheses}
\rightarrow
\text{shared realization gauge}
\rightarrow
\text{predictive-equivalence compression}
\rightarrow
Z_D
\]

如果 identity / orthogonal TP 都稳定，则支持：

\[
\boxed{
Z_D
\text{ 更接近 transformation law 的 interventional predictive equivalence class，
而不是某个特定 TP 坐标轴上的 pattern。}
}
\]

---

# 31. 一句话执行标准

\[
\boxed{
\text{在固定 }R_D=0\text{ 的 perfect-TP 世界中，
从随机 }M_{\max}=5\text{ mechanism bank 开始，
独立完成 family identity → coordinate → duplicate pruning → full TP；
并同时在 identity TP 与 orthogonal TP 下达到稳定 recovery。}
\]
