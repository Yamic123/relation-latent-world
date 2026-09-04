# ADP12 实验执行方案：Posterior Responsibility + Evidence-Based Candidate Elimination

## 0. 实验目标

ADP11 暴露了两个问题：

1. hard assignment 会过早把 family 分给单一 candidate，造成 candidate starvation 或 mechanism mixing；
2. ADP9 只能删除已经学成的 functional duplicate，无法删除 dead / under-supported candidate。

因此 ADP12 只增加两件事：

\[
\boxed{\text{hard assignment}\rightarrow\text{posterior responsibility}}
\]

以及：

\[
\boxed{\text{unsupported candidate}\rightarrow\text{evidence-based elimination}}
\]

核心问题：

> 能否把 candidate 看成 competing mechanism hypotheses，让多条 family evidence 逐渐决定每个 hypothesis 的责任与生死，而不是一开始就 hard argmin？

---

## 1. 本实验暂时不做什么

ADP12 只使用 Identity TP。

原因：ADP11 中 identity / orthogonal TP 没有真正进入优化路径，因此 TP invariance 留待后续独立实验。

固定：

```text
world
dataset
response-family construction
Mmax = 5
mechanism architecture
shared alpha -> v coordinate
ADP9 functional duplicate criterion
B1 solver
R_D = 0
```

---

## 2. 起点

沿用 ADP11 from-zero 设置：

```text
random Mmax=5 mechanism bank
random-balanced initial family assignment / responsibility
random candidate-level a,b
no ADP5/6 mechanism checkpoint
no GT pair
no GT mechanism label
no GT v
no M_real used by learner
```

\(M_{\rm real}=3\) 仅 evaluator 最终检查。

---

## 3. 对照组与主实验

每个 optimization seed 都运行四个版本：

| Variant | Assignment | Unsupported pruning | Duplicate pruning |
|---|---|---|---|
| H0 | hard argmin | no | ADP9 |
| H1 | soft posterior | no | ADP9 |
| H2 | hard argmin | evidence-based death | ADP9 |
| **H3** | **soft posterior** | **evidence-based death** | **ADP9** |

主实验：**H3**。

---

## 4. Posterior Responsibility

原方法：

\[
c_U=\arg\min_jE(U,j)
\]

改成：

\[
q_U(j)
=
\frac{
\pi_j\exp[-E(U,j)/\tau_t]
}{
\sum_k\pi_k\exp[-E(U,k)/\tau_t]
}
\]

其中：

- \(U\)：一个完整 response family；
- \(E(U,j)\)：candidate \(j\) 对整个 family 的 prediction cost；
- \(\pi_j\)：candidate prior，主实验固定均匀；
- \(\tau_t\)：随训练轮数降低的 temperature。

主实验不学习 \(\pi_j\)，避免把 count prior 与 assignment uncertainty 混在一起。

---

## 5. Family Cost

沿用 ADP11：

\[
E(U,j)
=
\frac1R\sum_r
D_S
\left(
d_{U,r},
\mathcal M_j
\left(
S_U,
a_j\tilde\alpha_{U,r}+b_j
\right)
\right)
\]

其中：

\[
v_{U,r}^{(j)}
=
a_j\tilde\alpha_{U,r}+b_j
\]

禁止恢复 per-point free \(v\)。

---

## 6. Temperature Schedule

固定 40 rounds。

每轮先计算 learner-visible cost gap：

\[
g_t
=
\operatorname{median}_U
\left(
E_{U,(2)}-E_{U,(1)}
\right)
\]

其中 \(E_{U,(1)},E_{U,(2)}\) 是 best / second-best cost。

定义：

\[
\tau_t
=
\kappa_t\max(g_t,\epsilon)
\]

schedule：

```text
round 0–9:   kappa = 4.0
round 10–19: kappa = 2.0
round 20–29: kappa = 1.0
round 30–39: kappa = 0.5
```

\[
\epsilon=10^{-8}
\]

目的：

```text
early: 保留 hypothesis competition
late: posterior 逐渐变尖
```

---

## 7. Soft-Responsibility Training Loss

训练 mechanism 时：

\[
L_{\rm mech}
=
\sum_U
\sum_j
q_U(j)E(U,j)
\]

对 \(q_U(j)\) 使用 stop-gradient。

也就是：

```text
E-step:
compute q from current candidate costs

M-step:
update M_j,a_j,b_j using q-weighted family loss
```

不加额外 entropy regularizer。

---

## 8. 主实验禁止额外正则

正式主实验禁止加入：

```text
candidate orthogonality
embedding decorrelation
entropy maximization/minimization
diversity loss
GT-count prior
```

目的是单独检验 posterior evidence accumulation 是否解决 starvation。

---

## 9. Effective Responsibility

定义 candidate \(j\) 在 round \(t\) 的 posterior support：

\[
N^{eff}_{j,t}
=
\sum_U q_{U,t}(j)
\]

归一化：

\[
r_{j,t}
=
\frac{N^{eff}_{j,t}}{N_{\rm family}}
\]

EMA：

\[
\bar r_{j,t}
=
0.8\bar r_{j,t-1}
+
0.2r_{j,t}
\]

注意：

\[
\boxed{
r_j\text{ 只用于筛选 unsupported hypothesis，不能直接决定删除。}
}
\]

---

## 10. Candidate Death：两阶段原则

### Stage 1：长期缺乏 posterior support

连续 \(K=5\) 个 round：

\[
\bar r_{j,t}<\rho_{\rm screen}
\]

第一版：

\[
\rho_{\rm screen}=0.01
\]

才进入 death test。

这只是 screening，不是删除条件。

### Stage 2：删除后 predictive evidence 不下降

临时移除 candidate \(j\)，不更新 mechanism 参数。

对所有 family：

\[
q^{-j}_U(k)
=
\frac{q_U(k)}
{\sum_{\ell\neq j}q_U(\ell)},
\qquad k\neq j
\]

重新评估：

```text
IID heldout
cross-state
cross-realization
cross-base
family-level prediction
```

---

## 11. Evidence-Based Death Score

定义：

\[
\Delta_j^d
=
L_{drop(j)}^d
-
L_{keep}^d
\]

其中：

\[
d\in\{iid,S,v,base\}
\]

主 gate：

```text
max_d Δ_j^d < 0.01
```

同时要求：

```text
任一 family subgroup 的 NRMSE increase < 0.02
```

第二条用于保护 rare-but-necessary mechanism。

---

## 12. Bootstrap Safety Check

对 heldout bases 做 200 次 bootstrap。

计算：

\[
CI_{95}^{upper}(\Delta_j^d)
\]

正式删除要求：

```text
所有 domain:
95% upper CI < 0.01
```

如果 CI 跨过阈值：

```text
保留 candidate
```

candidate death 的语义是：

\[
\boxed{
\text{“当前数据没有提供足够证据证明这个 hypothesis 必须存在。”}
}
\]

而不是“usage 低所以删”。

---

## 13. Candidate Death 的执行时机

禁止训练早期直接杀 candidate。

只有同时满足：

```text
round >= 20
candidate 已连续 5 rounds 低 posterior support
global heldout NRMSE 已进入稳定区
```

才允许执行 death test。

每轮最多删除 1 个 candidate。

删除后：

```text
冻结 mechanisms 1 round
重新计算 q
重新验证
再继续训练
```

---

## 14. Candidate Death 后的保护

正式删除 candidate \(j\) 后：

```text
remove j from alive set
renormalize q over alive candidates
do not merge parameters
do not average mechanisms
```

如果下一轮出现：

```text
heldout NRMSE increase > 0.01
或
某 subgroup increase > 0.02
```

则：

```text
rollback deletion
blacklist candidate for 5 rounds
```

---

## 15. 训练结束后的 Duplicate Pruning

40 rounds 后，对 surviving candidates 运行原 ADP9 graph：

```text
cross-state R² > 0.995
cross-realization R² > 0.995
cross-base R² > 0.995
direct replacement max ΔNRMSE < 0.01
```

然后：

```text
highest-confidence edge
→ hard prune one duplicate
→ freeze mechanism
→ reassign
→ structured validation
→ rebuild graph
```

停止条件：

\[
\boxed{\text{no valid duplicate edge}}
\]

---

## 16. 为什么需要两种 pruning

### Unsupported hypothesis

\[
\boxed{
\text{长期没有 posterior evidence 支持其存在}
}
\]

处理：

\[
\boxed{\text{evidence-based candidate death}}
\]

### Functional duplicate

\[
\boxed{
\text{两个 candidate 都学成了，但预测上等价}
}
\]

处理：

\[
\boxed{\text{ADP9 functional duplicate pruning}}
\]

两者不能混为一谈。

---

## 17. Z4 Identity / Coordinate Gate

40 rounds 后，evaluator-only：

```text
每个 GT mechanism:
best candidate functional R² > 0.95

所有主要 surviving candidate:
purity > 0.90

min affine-aligned realization R² > 0.95
median order accuracy > 0.95

heldout-realization NRMSE < 0.05
```

允许 functional duplicates。

不允许 major mixed candidate。

---

## 18. Final Count / Family Gate

unsupported death + duplicate pruning 完成后：

```text
stop reason = no valid unsupported candidate
              AND no valid duplicate edge

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

## 19. Full-TP B1

只有 final family/count gate PASS 才进入 B1。

固定：

```text
inner steps = 60
restarts = 3
chunk size = 1024
same lambda_P as ADP10/11
```

B1 PASS：

```text
IID NRMSE < 0.05
101 NRMSE < 0.10
participation micro-F1 > 0.90
min active instance-effect R² > 0.90
min functional R² > 0.90
inactive Z2 FPR < 0.10
```

---

## 20. 3-Seed Pilot

先运行：

```text
optimization seeds = 0,1,2
```

四个 variants：

```text
H0 hard / no death
H1 soft / no death
H2 hard / evidence death
H3 soft / evidence death
```

主要比较：

| Variant | Identity formation | Dead slots | Mixing | Final M | B1 |
|---|---|---|---|---|---|
| H0 |  |  |  |  |  |
| H1 |  |  |  |  |  |
| H2 |  |  |  |  |  |
| H3 |  |  |  |  |  |

---

## 21. Pilot Gate

H3 至少：

\[
\boxed{\ge2/3\text{ end-to-end PASS}}
\]

并且至少满足一项：

```text
相对 H0：
major mixing seed 数下降
或
unsupported/dead slot 导致的 count failure 下降
```

若 H3 仍为 0/3：

```text
STOP
```

---

## 22. 10-Seed Strong Test

Pilot PASS 后固定所有超参数：

```text
seeds = 0..9
```

Strong pass：

\[
\boxed{
H3\ge8/10\text{ end-to-end PASS}
}
\]

报告：

```text
H0 vs H3 end-to-end recovery
Z4 identity recovery
mixing rate
unsupported candidate death count
duplicate prune count
M_discovered histogram
B1 pass rate
```

---

## 23. Rare-Mechanism Safety Stress Test

必须做。

构造额外 dataset：

```text
保持 world 不变
将一个真实 mechanism 的 family occurrence 降到约 5%
```

learner 不知道哪个 mechanism 被降频。

目标：

\[
\boxed{
\text{证明 evidence-based death 不会把 rare-but-necessary mechanism 误删。}
}
\]

要求：

```text
rare mechanism 仍保留
functional R² > 0.90
对应 heldout subgroup prediction 不崩溃
false-death rate <= 1/10 seeds
```

如果 rare mechanism 经常被删：

\[
\boxed{\text{candidate death rule FAIL}}
\]

即使 balanced E0 上 count recovery 很好也不能接受。

---

## 24. Negative Controls

### NC-1：Usage-Only Death

```text
EMA responsibility < 1%
→ 直接删除
```

与 evidence-based death 对比。

### NC-2：Hard Assignment + Evidence Death

即 H2。

判断 count 问题是否主要只缺 death。

### NC-3：Soft Responsibility Only

即 H1。

判断 soft assignment 是否已足以减少 starvation。

### NC-4：Fixed High Temperature

```text
kappa = 4
```

全程不 anneal。

### NC-5：Immediate Hardening

```text
kappa = 0.1
```

从 round 0 开始近似 hard assignment。

---

## 25. 关键 Telemetry

每 round 保存：

```text
q(U,j)
posterior entropy per family

r_j
EMA r_j

candidate family hard-count
candidate purity              # evaluator-only
functional R²                 # evaluator-only

heldout NRMSE
cross-state NRMSE
cross-v NRMSE
cross-base NRMSE

coordinate a,b
affine realization R²
order accuracy

death-screen events
bootstrap CI
death commit / rollback

duplicate graph
duplicate prune events
```

---

## 26. 必须生成的诊断图

### A. Candidate responsibility trajectory

横轴：round

纵轴：

\[
r_{j,t}
\]

观察 candidate 是逐渐获得 evidence、逐渐死亡，还是长期竞争。

### B. Family posterior entropy

\[
H_U
=
-\sum_jq_U(j)\log q_U(j)
\]

检查：

```text
early 高 uncertainty
→ late posterior 集中
```

是否发生。

### C. Mixing vs posterior uncertainty

检查 mixed family 是否对应高 entropy，而不是被 hard assignment 强行确定。

---

## 27. 失败分类

### F1：Soft collapse
所有 candidate 长期获得相似 responsibility，无法形成 identity。

### F2：Premature death
必要 mechanism 在成熟前被删除。

### F3：Rare mechanism false death
balanced 成功，但 rare stress 误删真实机制。

### F4：Persistent mixing
posterior 到后期仍无法集中，major candidate purity 低。

### F5：Unsupported slot remains
candidate 长期无 evidence，但 safety test始终无法通过。

### F6：Duplicate remains
candidate 都学成，但 ADP9 graph 未删除重复机制。

### F7：B1 failure
family-level mechanisms 正确，但 full TP composition 失败。

---

## 28. 输出目录

```text
outputs/
└── e0_adp12_bayesian_responsibility/
    ├── h0_hard_no_death/
    ├── h1_soft_no_death/
    ├── h2_hard_evidence_death/
    ├── h3_soft_evidence_death/
    │   ├── seed_000/
    │   │   ├── training/
    │   │   ├── responsibility/
    │   │   ├── candidate_death/
    │   │   ├── duplicate_pruning/
    │   │   ├── final/
    │   │   ├── b1/
    │   │   └── final_verdict.json
    │   └── ...
    ├── rare_mechanism_stress/
    ├── negative_controls/
    ├── aggregate_3seed/
    └── aggregate_10seed/
```

---

## 29. 每个 Run Summary

```json
{
  "variant": "H3",
  "seed": 0,
  "z4_identity_pass": true,
  "death_events": 1,
  "death_rollbacks": 0,
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

## 30. 推荐命令模板

```bash
# 3-seed pilot
python run_adp12.py --variant H0 --seed 0
python run_adp12.py --variant H1 --seed 0
python run_adp12.py --variant H2 --seed 0
python run_adp12.py --variant H3 --seed 0

python run_adp12.py --stage pilot_all --seeds 0,1,2
python run_adp12.py --stage aggregate --seeds 0,1,2

# Pilot PASS 后
python run_adp12.py --stage strong_test --seeds 0,1,2,3,4,5,6,7,8,9

# Rare mechanism stress
python run_adp12.py --stage rare_stress --variant H3 --seeds 0,1,2,3,4,5,6,7,8,9
```

---

## 31. 关键科学判定

### Case A：H3 明显优于 H0，且 rare stress PASS

支持：

\[
\boxed{
\text{posterior responsibility + predictive-evidence death
能够解决 hard assignment starvation，
并在不误删 rare mechanism 的情况下改善 global count recovery。}
}
\]

### Case B：H2≈H3，二者都优于 H0

说明主要问题是：

\[
\boxed{\text{缺少 unsupported candidate elimination}}
\]

soft responsibility 不是必要条件。

### Case C：H1 明显优于 H0，但 H2 改善有限

说明主要问题更接近：

\[
\boxed{\text{early hard assignment 导致的 starvation / premature commitment}}
\]

### Case D：H3 balanced PASS，但 rare stress FAIL

说明：

\[
\boxed{
\text{当前 death rule 仍然混淆 rare real mechanism 与 unsupported hypothesis。}
}
\]

不能接受。

---

## 32. 与 Bayesian 理论的对应

ADP12 不做全面 Bayesian neural network，只验证三个核心思想：

\[
\boxed{
1.\ \text{mechanism identity 是 hypothesis uncertainty，而不是立即 hard label}
}
\]

\[
\boxed{
2.\ \text{candidate 是否存在，应由累计 evidence 支持，而不是一次 usage 决定}
}
\]

\[
\boxed{
3.\ \text{global count 是 model selection：
unsupported hypothesis death + duplicate hypothesis compression}
}
\]

若实验成功，当前学习链更新为：

\[
\boxed{
\text{response family}
\rightarrow
\text{posterior mechanism competition}
\rightarrow
\text{shared realization coordinate}
\rightarrow
\text{evidence-supported identities}
\rightarrow
\text{unsupported hypothesis elimination}
\rightarrow
\text{duplicate compression}
\rightarrow
Z_D
}
\]

---

## 33. 一句话执行标准

\[
\boxed{
\text{不要让一次 hard argmin 决定 mechanism identity；
让多个 response-family evidence 累积成 posterior responsibility，
再用 heldout predictive evidence 决定哪些 hypothesis 应当死亡。}
\]
