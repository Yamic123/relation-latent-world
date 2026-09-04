# ADP13 实验执行方案：Sparse Posterior Competition + Predictive-Evidence Model Selection

## 0. 实验定位

ADP12 已经给出两个明确结果：

1. Soft-all posterior 失败：5 个 candidate 长期获得非零 responsibility，导致 distributed co-adaptation，unsupported-candidate death 无法触发；
2. Evidence-based death 本身有效：在 hard assignment 下，真正零责任的 candidate 可以被安全删除，其中 H2 seed 2 成功完成 5→4→3 并通过 B1。

因此 ADP13 改为：

\[
oxed{	ext{Sparse posterior hypothesis competition}+	ext{predictive-evidence model selection}}
\]

核心思想：

> 保留“identity 仍有不确定性”这一 Bayesian 思想，但不允许所有 candidate 都持续获得训练 credit；candidate 是否必要，也不再由 responsibility 大小决定，而由“移除它是否损害 held-out intervention prediction”决定。

---

## 1. 核心科学问题

### Q1：Sparse posterior 是否能解决 Soft-all 的 distributed co-adaptation？

\[
oxed{	ext{允许少数 hypothesis 竞争，而不是让 5 个 hypothesis 一起训练}}
\]

### Q2：Candidate existence 是否可以由 predictive evidence 决定？

\[
oxed{	ext{一个 candidate 是否应存在，取决于移除它是否损害未来预测}}
\]

---

## 2. 固定项

沿用 ADP11/12：

```text
E0 synthetic world
R_D = 0
Identity TP
same world/data/family seeds
same train/val/test splits
Mmax = 5
same response-family construction
same mechanism architecture
same shared alpha -> v coordinate
same 40-round training horizon
same optimizer/LR/batch
same ADP9 duplicate-pruning gate
same B1 solver
```

禁止 learner 使用：

```text
GT mechanism label
GT v
GT participation
GT duplicate pair
M_real = 3
```

\(M_{real}=3\) 仅 evaluator 最终检查。

---

## 3. 实验变体

历史结果作为基线：

| Variant | Assignment | Existence test | 状态 |
|---|---|---|---|
| H0 | hard argmin | none | 复用 ADP11 |
| H2 | hard argmin | low-resp screen + evidence death | 复用 ADP12 |
| H3 | soft-all | low-resp screen + evidence death | 复用 ADP12 |

ADP13 新运行：

| Variant | Assignment | Existence test | 作用 |
|---|---|---|---|
| S1 | sparse posterior top-2 | none | 单独测试 sparse competition |
| **S2** | **sparse posterior top-2** | **predictive-evidence test** | **主实验** |
| S3 | sparse posterior top-3 | predictive-evidence test | sparsity 强度 control |
| S4 | sparse posterior top-1 | predictive-evidence test | 近 hard control |

---

## 4. From-Zero 起点

每个新 run：

```text
random Mmax=5 mechanism bank
random-balanced initial family assignment
random candidate-level a,b
no ADP5/6 checkpoint
no precomputed duplicate graph
no warm-start mechanism
```

同一 optimization seed 下 S1/S2/S3/S4 使用相同初始化。

---

## 5. Family Cost

\[
E(U,j)=
rac1R\sum_r
D_S
\left(
d_{U,r},
\mathcal M_j
\left(
S_U,
a_j	ildelpha_{U,r}+b_j
ight)
ight)
\]

并保持：

\[
v_{U,r}^{(j)}=a_j	ildelpha_{U,r}+b_j
\]

禁止恢复 per-point free \(v\)。

---

## 6. Full Posterior

先计算：

\[
	ilde q_U(j)
=
rac{\exp[-E(U,j)/	au_t]}
{\sum_k\exp[-E(U,k)/	au_t]}
\]

固定 uniform prior：

\[
\pi_j=rac1{M_{alive}}
\]

ADP13 不同时引入 learned Dirichlet prior。

---

## 7. Sparse Posterior

对每个 family \(U\)，取 full posterior 最大的 top-\(K\) candidates：

\[
\mathcal T_K(U)=TopK_j\{	ilde q_U(j)\}
\]

定义：

\[
q_U^{(K)}(j)=
egin{cases}
\dfrac{	ilde q_U(j)}
{\sum_{k\in\mathcal T_K(U)}	ilde q_U(k)},
& j\in\mathcal T_K(U)\
0,& 	ext{otherwise}
\end{cases}
\]

主实验：

\[
oxed{K=2}
\]

解释：一个 family 可以暂时在两个 competing hypotheses 之间保持不确定性，但其余 candidates 不获得该 family 的训练 credit。

---

## 8. Temperature Schedule

沿用 ADP12：

\[
g_t=
\operatorname{median}_U
\left(
E_{U,(2)}-E_{U,(1)}
ight)
\]

\[
	au_t=
\kappa_t\max(g_t,10^{-8})
\]

```text
round 0–9:   kappa = 4.0
round 10–19: kappa = 2.0
round 20–29: kappa = 1.0
round 30–39: kappa = 0.5
```

---

## 9. Sparse-Posterior Training

每轮：

```text
E-step:
1. compute E(U,j)
2. compute full posterior q_tilde
3. truncate to top-K
4. renormalize to sparse q

M-step:
5. stop-gradient q
6. update M_j, a_j, b_j using q-weighted loss
```

训练 loss：

\[
L_{mech}
=
\sum_U
\sum_{j\in\mathcal T_K(U)}
q_U^{(K)}(j)E(U,j)
\]

禁止：

```text
entropy regularizer
orthogonality
decorrelation
diversity loss
GT-count penalty
```

---

## 10. Candidate Existence：不再依赖 Low-Responsibility Screen

ADP12 的：

```text
EMA responsibility < 1%
```

从正式 death 条件中完全删除。

从 round 20 开始，每 5 rounds 对所有 alive candidates 做一次 existence test：

```text
round 20
round 25
round 30
round 35
round 40/final
```

---

## 11. Candidate 的两个假设

对于 candidate \(j\)：

\[
H_j^{keep}:	ext{candidate }j	ext{ 是必要的独立 mechanism hypothesis}
\]

\[
H_j^{drop}:	ext{世界无需 candidate }j
\]

比较两者对 held-out intervention evidence 的预测能力。

---

## 12. Frozen Counterfactual Drop

临时移除 candidate \(j\)：

```text
mechanism parameters 不更新
其他 candidate 参数不更新
```

对每个 heldout family：

1. 删除 \(j\)；
2. 用剩余 candidate 重新计算 \(E(U,k)\)；
3. 重新形成 sparse posterior；
4. 计算 drop-model prediction。

测试的是：

\[
oxed{	ext{其他现有 hypotheses 能否在不重新学习机制函数的情况下解释原 evidence}}
\]

---

## 13. Predictive-Evidence Contribution

对：

\[
d\in\{IID,state,realization,base\}
\]

定义：

\[
\Delta_j^d
=
L_{drop(j)}^d-L_{keep}^d
\]

解释：

- \(\Delta_j^d\gg0\)：删除显著损害预测，candidate 有独立 evidence；
- \(\Delta_j^dpprox0\)：删除几乎不影响预测，candidate 缺乏独立 evidence。

---

## 14. Base-Level Evidence Distribution

对 heldout bases：

\[
\delta_{j,b}
=
L_{drop(j),b}
-
L_{keep,b}
\]

保存：

\[
\{\delta_{j,b}\}_{b=1}^{N_{base}}
\]

并做 500 次 bootstrap。

记录：

```text
mean delta
median delta
95% CI
95th percentile subgroup damage
```

---

## 15. Existence Gate

candidate \(j\) 只有同时满足：

```text
A. 所有 global domains:
   bootstrap upper 95% CI of ΔNRMSE < 0.01

B. 任一预定义 subgroup:
   ΔNRMSE < 0.02

C. 删除后每个 learner-visible heldout family
   都至少存在一个 remaining candidate 可形成有限预测

D. 删除后 global heldout NRMSE 不超过 keep model + 0.01
```

才判：

\[
oxed{	ext{unsupported / non-essential hypothesis}}
\]

注意：

\[
oxed{	ext{candidate responsibility 大小完全不进入正式 existence gate}}
\]

---

## 16. 每次最多删除一个 Candidate

若多个 candidate 均可删，定义：

\[
Score_j=
\max_d CI_{95}^{upper}(\Delta_j^d)
\]

选择：

\[
j^*=rg\min_j Score_j
\]

然后：

```text
hard delete
freeze mechanisms 1 round
recompute sparse posterior
structured validation
```

---

## 17. Rollback Gate

删除后若出现任何：

```text
global heldout NRMSE increase > 0.01
任一 subgroup NRMSE increase > 0.02
任何 GT evaluator functional identity best R² < 0.90
```

则：

```text
rollback
该 candidate 当前 round blacklist
```

GT functional identity 只用于实验安全审计，不参与 learner 决策。

---

## 18. ADP9 Duplicate Pruning

训练结束后仍执行 ADP9：

```text
cross-state R² > 0.995
cross-realization R² > 0.995
cross-base R² > 0.995
direct-replacement max ΔNRMSE < 0.01
```

学习链：

\[
oxed{
	ext{existence selection}
ightarrow
	ext{functional duplicate compression}
}
\]

---

## 19. Z4 Identity Gate

40 rounds 后 evaluator-only：

```text
每个 GT mechanism:
best candidate functional R² > 0.95

所有 major surviving candidates:
purity > 0.90

min affine-aligned realization R² > 0.95
median order accuracy > 0.95

heldout-realization NRMSE < 0.05
```

允许 duplicates。

---

## 20. Final Family / Count Gate

Existence selection + duplicate pruning 后：

```text
no valid existence deletion
no valid duplicate edge

family Hungarian accuracy > 0.90
mean fragmentation < 0.10
assignment change < 0.10

all matched functional R² > 0.95

IID NRMSE < 0.035
cross-state NRMSE < 0.05
cross-realization NRMSE < 0.05
cross-base NRMSE < 0.05
```

最后 evaluator 才检查：

```text
M_discovered == 3 ?
```

---

## 21. Full-TP B1

只有 final gate PASS 才进入 B1。

固定：

```text
inner steps = 60
restarts = 3
chunk size = 1024
same lambda_P
20k IID
20k combination-101
```

PASS：

```text
IID NRMSE < 0.05
101 NRMSE < 0.10
participation micro-F1 > 0.90
min active instance-effect R² > 0.90
min functional R² > 0.90
inactive Z2 FPR < 0.10
```

---

## 22. 3-Seed Pilot

新运行：

```text
S1 top2 / no existence selection
S2 top2 / predictive-evidence selection
S3 top3 / predictive-evidence selection
S4 top1 / predictive-evidence selection
```

seeds：

```text
0,1,2
```

共 12 个新 runs。

---

## 23. Pilot PASS

主实验 S2 必须：

\[
oxed{\ge2/3	ext{ end-to-end PASS}}
\]

且：

```text
Z4 recovery >= 2/3
major mixing seeds <= 1/3
没有 evidence-selection false deletion
```

若 S2 为 0/3 或 1/3：

```text
STOP
```

不扩 10 seeds。

---

## 24. Top-K 对照解释

### 若 S1 比 H3 明显好

说明 sparse hypothesis competition 本身解决了 soft-all co-adaptation。

### 若 S1 identity 好但 Final M>3，而 S2 达到 M=3

说明：

\[
oxed{
	ext{sparse posterior 解决 identity；
predictive-evidence selection 解决 global count}
}
\]

### 若 S4(top1) ≈ H2

说明 top1 确实接近 hard assignment。

### 若 S2(top2) > S4(top1)

支持：

\[
oxed{	ext{保留少量 posterior uncertainty 有实际价值}}
\]

### 若 S3(top3) 明显差于 S2

支持：

\[
oxed{	ext{过宽 posterior support 会重新引入 distributed co-adaptation}}
\]

---

## 25. 10-Seed Strong Test

只有 S2 pilot PASS 后：

```text
S2 seeds = 0..9
```

strong pass：

\[
oxed{\ge8/10	ext{ end-to-end PASS}}
\]

报告：

```text
Z4 recovery rate
mixing rate
existence-deletion count
rollback count
duplicate-prune count
M_discovered histogram
B1 pass rate
```

---

## 26. Rare-Mechanism Safety Stress

仅在 S2 strong-test ≥8/10 后执行。

构造：

```text
一个真实 mechanism family frequency ≈ 5%
其他 world mechanism 不变
```

要求：

```text
rare mechanism survival >= 9/10
rare mechanism functional R² > 0.90
rare subgroup heldout prediction PASS
false-deletion <= 1/10
```

若 balanced E0 成功但 rare stress 经常误删：

\[
oxed{	ext{predictive-evidence existence rule FAIL}}
\]

---

## 27. Optional Bayesian Extension：Dirichlet Prior

仅当：

```text
S2 identity recovery 良好
但仍存在多个长期 weak candidates
且 predictive-evidence selection 不稳定
```

才单独开后续子实验：

\[
\pi\sim Dirichlet(lpha_0,\ldots,lpha_0)
\]

测试：

```text
alpha0 = 1.0
alpha0 = 0.5
alpha0 = 0.2
```

不属于 ADP13 主实验。

---

## 28. 必须保存的 Telemetry

每 round：

```text
E(U,j)
full q_tilde(U,j)
sparse q(U,j)
top-K membership

full entropy
sparse entropy
top1 mass
top2 mass
top1-top2 margin

candidate weighted responsibility
candidate top-K inclusion frequency

candidate purity              # evaluator-only
functional R² matrix          # evaluator-only

heldout NRMSE
cross-state NRMSE
cross-v NRMSE
cross-base NRMSE

coordinate a,b
affine realization R²
order accuracy
```

每次 existence test：

```text
candidate
keep losses
drop losses
base-level delta distribution
bootstrap CI
subgroup max delta
decision
rollback result
```

---

## 29. 必须生成的诊断图

### Figure A：Top-K membership trajectory

每个 candidate：

\[
P(j\in\mathcal T_K(U))
\]

随 round 变化。

### Figure B：Full vs Sparse Entropy

比较：

\[
H^{full}\quad vs\quad H^{sparse}
\]

### Figure C：Candidate Predictive Evidence

每次 existence test 绘制：

\[
CI_{95}(\Delta_j)
\]

### Figure D：Mixing vs Top-K overlap

检查 mixed candidate 是否来自长期共享同一批 families。

---

## 30. 失败分类

### F1：Sparse co-adaptation
top-2 仍共同拟合同一 mechanism family。

### F2：Premature exclusion
真实机制 candidate 早期不进入 top-K，之后无法恢复。

### F3：Evidence false death
candidate 被删除后发现其对应必要机制。

### F4：Evidence indecision
明显冗余 candidate drop impact 始终不够小。

### F5：Persistent mixing
40 rounds 后 major candidate purity 仍低。

### F6：Duplicate remains
成熟等价 candidate 未被 ADP9 压缩。

### F7：Count-only failure
identity 正确，但 \(M_{discovered}
eq3\)。

### F8：B1 failure
最终 mechanisms 正确，但 full-TP composition 失败。

---

## 31. 输出目录

```text
outputs/
└── e0_adp13_sparse_bayesian/
    ├── s1_top2_no_evidence/
    ├── s2_top2_evidence/
    │   ├── seed_000/
    │   │   ├── training/
    │   │   ├── posterior/
    │   │   ├── existence_tests/
    │   │   ├── duplicate_pruning/
    │   │   ├── final/
    │   │   ├── b1/
    │   │   └── final_verdict.json
    │   └── ...
    ├── s3_top3_evidence/
    ├── s4_top1_evidence/
    ├── aggregate_3seed/
    ├── aggregate_10seed/
    └── rare_stress/
```

---

## 32. 每个 Run Summary

```json
{
  "variant": "S2",
  "seed": 0,
  "K": 2,

  "z4_identity_pass": true,
  "major_mixing": false,

  "existence_tests": 15,
  "existence_deletes": 1,
  "existence_rollbacks": 0,

  "duplicate_prunes": 1,

  "m_discovered": 3,

  "family_accuracy": 1.0,
  "fragmentation": 0.0,

  "b1_pass": true,
  "final_pass": true,
  "failure_mode": null
}
```

---

## 33. 命令模板

```bash
# Smoke test
python run_adp13.py --stage smoke

# 3-seed pilot
python run_adp13.py --stage pilot_all --seeds 0,1,2

# 单独复现
python run_adp13.py --variant S2 --seed 0

# 汇总
python run_adp13.py --stage aggregate --seeds 0,1,2

# Pilot PASS 后
python run_adp13.py --stage strong_test --variant S2 --seeds 0,1,2,3,4,5,6,7,8,9

# Strong PASS 后
python run_adp13.py --stage rare_stress --variant S2 --seeds 0,1,2,3,4,5,6,7,8,9
```

---

## 34. 禁止的事后操作

正式 pilot 开始后禁止：

```text
看到 S2 某 seed 失败后改 K
只给失败 seed 增加 rounds
只给失败 seed调 temperature
看到 M=4 后放宽 existence threshold
使用 M_real=3 强制继续 prune
看到 B1 near-miss 后单独增加 restarts
```

任何修改必须成为 ADP14。

---

## 35. 最重要的科学判定

### Case A：S2 ≥2/3，且明显优于 H2/H3

支持：

\[
oxed{
	ext{有限 posterior uncertainty + sparse credit assignment
比 hard commitment 或 soft-all 更适合 from-zero mechanism identity formation}
}
\]

如果同时 existence selection 恢复正确 count，则进一步支持：

\[
oxed{	ext{global mechanism count 可以由 predictive evidence 而非 usage 决定}}
\]

### Case B：S1 identity 好，S2 count 好

最理想分解：

\[
oxed{
	ext{sparse posterior 解决 identity；
predictive-evidence selection 解决 existence/count}
}
\]

### Case C：S2 与 H2 相近

说明 Bayesian uncertainty 没有额外价值，主要有效模块仍是 evidence-based candidate elimination。

### Case D：S2 仍出现 distributed mixing

说明 top-2 仍然过宽，应研究更结构化 posterior hypothesis representation，而不是继续调 death threshold。

### Case E：Top-2 经常 premature exclusion

说明 sparse posterior 太早截断 hypothesis space，应研究 delayed sparsification、candidate revival 或 sequential Bayesian updating。

---

## 36. 与 Bayesian 理论的对应

ADP13 对应的 Bayesian 解释：

\[
oxed{
	ext{uncertainty 应存在于少数真正 competing hypotheses 之间，
而不是平均扩散给所有 hypotheses}
}
\]

同时：

\[
oxed{
	ext{一个 hypothesis 是否存在，
应由它对 held-out intervention prediction 的独立贡献决定}
}
\]

学习链：

\[
oxed{
	ext{response family}
ightarrow
	ext{sparse posterior hypothesis competition}
ightarrow
	ext{shared realization coordinate}
ightarrow
	ext{predictive-evidence existence selection}
ightarrow
	ext{functional duplicate compression}
ightarrow
Z_D
}
\]

---

## 37. 一句话执行标准

\[
oxed{
	ext{允许一个 family 暂时在少数机制假设间保持不确定，
但只给真正竞争者训练 credit；
然后不看 usage，而看移除某 hypothesis 是否损害未来预测，
来决定它是否应当存在。}
\]
