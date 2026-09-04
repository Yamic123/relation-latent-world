# ADP10 实验执行方案：从 ADP6 开始验证后半链路稳定性

## 0. 实验定位

ADP9 在 development seed 0 上已经闭合：

\[
5 \rightarrow 4 \rightarrow 3
\]

并且最终：

```text
family Hungarian accuracy = 1.0
mean fragmentation = 0.0
三个 functional R² > 0.99988
full-TP B1 PASS
```

但当前只有一个独立的 ADP6-A2 seed-0 checkpoint，因此目前只能说明：

\[
\boxed{
\text{ADP6→ADP9 在 seed 0 上可行}
}
\]

还不能说明：

\[
\boxed{
\text{ADP6→ADP9 这条后半链路是稳定的}
}
\]

ADP10 的目标是：

\[
\boxed{
\text{固定上游 family-level evidence，
从 ADP6 开始生成多个独立 optimization seed，
检验 coordinate formation → duplicate pruning → full TP composition 是否稳定。}
}
\]

---

## 1. 本实验能证明什么

如果 ADP10 多 seed 成功，可以支持：

\[
\boxed{
\text{在 ADP5 已形成可用 response family identity 的条件下，
ADP6→ADP9 后半链路是稳定的。}
}
\]

验证内容：

```text
realization coordinate formation
functional identity recovery
functional duplicate identification
gradual duplicate pruning
global mechanism count recovery
full TP participation/composition
```

但仍不能证明：

\[
\boxed{
\text{整套系统从最随机 candidate bank 开始就是稳定的。}
}
\]

这个更强结论留给后续 from-zero 实验。

---

## 2. 固定与变化的因素

### 2.1 固定

所有 seed 共用：

```text
WORLD_SEED = 20260901
DATASET_SEED = 20260902

same E0 world
same train/val/test split
same response families
same observable alpha construction
same Mmax = 5
same mechanism architecture
same ADP6 / ADP9 hyperparameters
same stage gates
same B1 solver configuration
```

### 2.2 变化

只改变：

```text
optimization_seed
model parameter initialization randomness
minibatch order
optimizer stochasticity
tie-breaking randomness（如存在）
```

第一阶段：

```text
seeds = {0,1,2}
```

正式阶段：

```text
seeds = {0,1,2,...,9}
```

---

## 3. 起点定义

所有 seed 使用同一份 learner-visible ADP5 输入：

```text
response family dataset
family assignment / family identity evidence
observable alpha
```

但从 ADP6 的 mechanism-coordinate learning 开始，各 seed 独立运行。

每个 seed 都必须独立产生：

```text
ADP6-A1 checkpoint
ADP6-A2 checkpoint
ADP9 equivalence graph
ADP9 pruning path
ADP9 final pruned model
ADP9 B1 result
```

禁止：

```text
seed1/seed2 直接复制 seed0 ADP6-A2 checkpoint
只修改 ADP9 RNG 后称为独立 seed
使用 seed0 的 duplicate pair 结果硬编码 seed1/seed2
使用 M_real=3 控制停止
```

---

## 4. 总体流程

对每个 optimization seed \(s\)：

```text
ADP10-S0  输入与随机性审计
   ↓
ADP6-A1   fixed-family structured realization coordinate
   ↓
ADP6-A2   joint family assignment + coordinate
   ↓
ADP10-S1  ADP6 functional stability gate
   ↓
ADP9-0    自动 functional equivalence graph
   ↓
ADP9      gradual prune until no duplicate edge
   ↓
ADP10-S2  count / identity stability gate
   ↓
ADP9-B1   full TP participation
   ↓
ADP10-S3  downstream end-to-end verdict
```

任何 seed 在正式 gate FAIL：

```text
该 seed 记为 FAIL
保存失败阶段
不人为修复后继续算作 PASS
```

可另外开 diagnostic run，但必须与正式 run 分开。

---

## 5. ADP10-S0：随机性与独立性审计

对 seed 0/1/2 保存：

```text
optimization_seed
initial parameter hash
initial optimizer state hash
data split hash
family dataset hash
config hash
code commit
```

要求：

```text
dataset/world hashes 完全相同
initial model hashes 不同
optimization_seed 不同
```

PASS 条件：

\[
\boxed{
\text{同一科学问题 + 独立优化轨迹}
}
\]

---

## 6. ADP6-A1：固定 family identity，重新形成 realization coordinate

沿用 ADP6：

\[
v_{U,r}
=
a_{c_U}\tilde\alpha_{U,r}
+
b_{c_U}
\]

固定 family assignment，只学习：

```text
mechanism function
candidate-level shared affine coordinate
```

禁止：

```text
per-point free v
GT amplitude supervision
GT mechanism label
```

---

## 7. ADP6-A1 指标与 gate

每个 seed 记录：

```text
heldout-realization NRMSE

functional R² matrix
matched functional R²

candidate affine-aligned realization R²
order accuracy
same-alpha cross-family variance

candidate a,b
```

PASS：

```text
median order accuracy > 0.95
min affine-aligned realization R² > 0.90
heldout-realization NRMSE < 0.15

至少两个 purity-high candidate functional R² > 0.70
matched functional R² 不能全部为负
```

---

## 8. ADP6-A2：恢复 assignment learning

对 family：

\[
E(U,j)
=
\frac1R
\sum_r
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

然后：

\[
c_U=\arg\min_j E(U,j)
\]

交替：

```text
assign
update M_j,a_j,b_j
evaluate
```

不做 merge，不做 global count。

---

## 9. ADP10 对 A2 的稳定性判据

seed 0 已知 duplicate symmetry 会导致 assignment 振荡，因此本实验不把旧的 assignment-stability gate 当作 ADP6 是否成功的核心条件。

A2 核心 gate：

```text
三个 evaluator GT mechanism 各自 best functional R² > 0.95
所有 purity-high candidate functional R² > 0.90

order accuracy > 0.95
affine realization R² > 0.95
heldout-realization NRMSE < 0.05

candidate purity 保持高
不出现明显 mechanism mixing
```

允许：

```text
duplicate candidates 之间 family assignment 振荡
```

不允许：

```text
真实 mechanism functional identity 丢失
realization coordinate 失效
candidate 变成混合机制
```

---

## 10. ADP10-S1：ADP6 稳定性判定

汇总：

| Seed | A1 | A2 functional | Coordinate | Mixing | 进入 ADP9? |
|---|---|---|---|---|---|
| 0 |  |  |  |  |  |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |

第一阶段要求：

\[
\boxed{
\ge2/3
}
\]

seed 满足 ADP6 functional gate。

若仅 1/3：

```text
STOP
```

说明瓶颈仍在 ADP6 coordinate / identity formation。

---

## 11. ADP9-0：每个 seed 独立构建功能等价图

对每个通过 S1 的 seed，两两计算：

```text
cross-state prediction R²
cross-realization prediction R²
cross-base prediction R²
direct-replacement max ΔNRMSE
```

duplicate edge gate：

```text
cross-state R² > 0.995
cross-realization R² > 0.995
cross-base R² > 0.995
direct-replacement max ΔNRMSE < 0.01
```

禁止从 seed 0 复制 candidate pair。

---

## 12. 跨 seed 比较必须 permutation-invariant

candidate index 没有跨 seed 固定语义。

因此 seed 1 的 `C1` 不需要对应 seed 0 的 `C1`。

跨 seed 比较必须基于：

```text
functional matching
Hungarian matching
equivalence class
```

而不是 candidate 编号。

记录每个 seed 的 functional partition：

\[
\Pi_s
\]

以及最终：

\[
|\Pi_s|
\]

---

## 13. ADP9 gradual prune

完全沿用 ADP9：

```text
构图
↓
选最高置信 duplicate edge
↓
选择 learner-visible survivor
↓
hard prune
↓
冻结 mechanism
↓
全量 reassignment
↓
重新验证
↓
重新构图
```

停止条件：

\[
\boxed{
\text{no valid duplicate edge}
}
\]

禁止：

```text
alive_count == 3 时强制停止
```

---

## 14. 每个 seed 必须保存 pruning path

允许结果：

```text
5 -> 4 -> 3 -> stop
5 -> 4 -> stop
5 -> 4 -> 3 -> 2 -> stop
```

不得人工修正。

---

## 15. ADP10-S2：global mechanism count stability

每个 seed pruning 后记录：

```text
M_discovered
stop reason
functional classes
final family accuracy
final fragmentation
final assignment stability
functional R²
structured NRMSE
```

单 seed pruning PASS：

```text
stop reason = no valid duplicate edge

三个 evaluator GT mechanism 均有对应 functional identity
matched functional R² > 0.95

family Hungarian accuracy > 0.90
fragmentation < 0.10
assignment change < 0.10
```

最后 evaluator 才记录：

```text
M_discovered == 3 ?
```

---

## 16. 第一阶段 count stability

要求：

\[
\boxed{
\text{至少 2/3 seeds 自动得到 }M_{\rm discovered}=3
}
\]

并且对应 seed：

```text
functional gate PASS
structured holdout PASS
family identity PASS
```

不能只看 count=3。

---

## 17. ADP9-B1：full TP composition

每个通过 pruning 的 seed 都使用自己的 surviving mechanisms 独立执行 B1。

枚举：

\[
2^{M_{\rm discovered}}
\]

个 support。

优化：

\[
\alpha_i^j
\]

并使用：

\[
v_i^j=a_j\alpha_i^j+b_j.
\]

---

## 18. B1 solver 固定

一开始固定：

```text
inner steps = 60
restarts = 3
```

所有 seed 相同。

禁止某个 seed 失败后单独增加求解预算并仍算正式 PASS。

---

## 19. B1 gate

```text
IID NRMSE < 0.05
101 NRMSE < 0.10

participation micro-F1 > 0.90

min active instance-effect R² > 0.90
min functional R² > 0.90

inactive Z2 FPR on 101 < 0.10
```

---

## 20. ADP10-S3：第一阶段最终判定

单 seed PASS：

```text
ADP6-A1 PASS
+
ADP6-A2 functional PASS
+
ADP9 automatic pruning PASS
+
M_discovered = 3        # evaluator-only final check
+
B1 PASS
```

三 seed 第一阶段 PASS：

\[
\boxed{
\ge2/3
}
\]

---

## 21. 通过后扩到 10 seeds

若 2/3 PASS：

```text
seeds = 0..9
```

不修改：

```text
threshold
hyperparameter
dataset
world
solver
duplicate gate
```

Strong pass：

\[
\boxed{
\ge8/10
}
\]

---

## 22. 10-seed 必须报告

```text
ADP6-A1 pass rate
ADP6-A2 functional pass rate
ADP9 count recovery rate
B1 pass rate
downstream end-to-end pass rate
```

并报告分布：

```text
M_discovered histogram
prune step count histogram
functional R² distribution
IID NRMSE distribution
101 NRMSE distribution
participation F1 distribution
```

---

## 23. 失败分类

### F1：ADP6-A1 coordinate failure

```text
order / affine coordinate FAIL
functional R² 低
```

### F2：ADP6-A2 mechanism mixing

```text
candidate purity 低
functional identity 无法匹配
```

### F3：duplicate graph false negative

```text
仍有明显冗余
但 graph 不建边
M_discovered > 3
```

### F4：duplicate graph false positive

```text
错误机制被判 duplicate
prune 后 functional R² 崩溃
```

### F5：pruning / reassignment instability

```text
pair 可安全替代
但删除后 assignment 不稳定
```

### F6：full TP inference failure

```text
ZD 正确
但 support inference / composition 失败
```

---

## 24. Negative controls

至少保留两个跨 seed control。

### NC-1：usage prune

对 seeds 0/1/2：

```text
按 usage 从低到高删
```

比较其跨 seed 稳定性。

### NC-2：joint fine-tuning after prune

对每个 seed 的第一条 duplicate edge：

```text
direct prune
vs
prune + mechanism fine-tuning
```

检查 ADP8 的 catastrophic drift 是否跨 seed 重现。

---

## 25. 输出目录

```text
outputs/
└── e0_adp10_from_adp6/
    ├── seed_000/
    │   ├── s0_audit/
    │   ├── adp6_a1/
    │   ├── adp6_a2/
    │   ├── adp9_graph/
    │   ├── adp9_pruning/
    │   ├── adp9_b1/
    │   └── final_verdict.json
    ├── seed_001/
    ├── seed_002/
    ├── aggregate_3seed/
    │   ├── summary.csv
    │   ├── failure_modes.json
    │   └── final_verdict.json
    └── aggregate_10seed/
```

---

## 26. 每个 seed 的最终 summary

```json
{
  "seed": 0,
  "adp6_a1_pass": true,
  "adp6_a2_functional_pass": true,
  "adp9_prune_pass": true,
  "m_discovered": 3,
  "b1_pass": true,
  "downstream_end_to_end_pass": true,
  "failure_mode": null
}
```

---

## 27. 三 seed 汇总表

| Seed | A1 coordinate | A2 functional | Final M | Prune stable | B1 IID | B1 101 | F1 | Final |
|---|---|---|---:|---|---:|---:|---:|---|
| 0 |  |  |  |  |  |  |  |  |
| 1 |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |

---

## 28. 推荐命令模板

```bash
python run_adp10.py --stage seed_run --seed 0
python run_adp10.py --stage seed_run --seed 1
python run_adp10.py --stage seed_run --seed 2

python run_adp10.py --stage aggregate --seeds 0,1,2

# 2/3 PASS 后
python run_adp10.py --stage seed_run --seed 3
python run_adp10.py --stage seed_run --seed 4
python run_adp10.py --stage seed_run --seed 5
python run_adp10.py --stage seed_run --seed 6
python run_adp10.py --stage seed_run --seed 7
python run_adp10.py --stage seed_run --seed 8
python run_adp10.py --stage seed_run --seed 9

python run_adp10.py --stage aggregate --seeds 0,1,2,3,4,5,6,7,8,9
```

---

## 29. 第一阶段成功后的下一实验

若：

\[
\boxed{
\ge2/3
}
\]

从 ADP6 开始稳定通过，则下一步再做：

\[
\boxed{
\text{from-zero stability under perfect TP}
}
\]

并加入：

\[
\boxed{
\text{identity TP vs orthogonal TP}
}
\]

对照。

---

## 30. 科学结论的边界

如果 ADP10 成功，可以说：

\[
\boxed{
\text{在已有 response-family identity evidence 的条件下，
统一 realization coordinate + functional duplicate pruning + full TP composition
具有优化稳定性。}
}
\]

仍不能说：

\[
\boxed{
\text{完整机制发现系统从随机初始化开始已经稳定。}
}
\]

---

## 31. 一句话执行标准

\[
\boxed{
\text{固定同一理想数据与 family evidence，
仅改变 optimization seed，
独立重跑 ADP6→ADP9→B1；
若至少 2/3，再扩展到 8/10，
才认为后半机制发现链稳定。}
\]
