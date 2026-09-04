# ADP14 实验执行方案：Constrained Specialist Absorption

## 0. 实验定位

ADP13-S4 当前是最强路线：

\[
\boxed{\text{top-1 assignment + proactive predictive-evidence selection}}
\]

其 3-seed pilot 中 seed 0/2 end-to-end PASS，seed 1 失败。

对 seed 1 的逐轮 assignment / cost 与 round 35/39 existence tests 的诊断表明：失败并不是简单存在“完全无用”的 candidate，而更像训练路径把同一个全局 mechanism 裂成了 **core mechanism + context specialist**。

当前 frozen-delete test 问的是：

> “现在直接拔掉 candidate \(j\)，其他参数不变，预测是否变差？”

但这不能回答：

> “如果世界里本来没有 \(j\)，现有 parent mechanism 能否通过有限、受约束的再适应吸收 \(j\) 的 contexts？”

ADP14 专门验证这个问题。

核心假设：

\[
\boxed{
\text{early context partition}
\rightarrow
\text{gradient partition}
\rightarrow
\text{specialization}
\rightarrow
\text{self-created predictive necessity}
}
\]

如果一个所谓 specialist 能被另一个 candidate 在保持其原有功能的情况下吸收，则它更可能是 optimization-created fragmentation，而不是独立 \(Z_D\)。

---

# 1. 核心科学问题

ADP14 只回答一个问题：

\[
\boxed{
\text{失败 seed 中的额外 candidates 是否是可被已有 mechanism 吸收的 context specialists？}
}
\]

不是测试：

- 新的 soft posterior；
- 新的 TP 表征；
- 新的 \(R_D\)；
- 新的 mechanism architecture；
- 更宽松的 pruning threshold。

---

# 2. 实验起点

主诊断只使用：

```text
ADP13
variant = S4
seed = 1
round = 39 / final pre-prune checkpoint
```

原因：

- seed0/2 已经成功；
- seed1 是当前唯一 persistent fragmentation / mixing failure；
- 现有 existence tests 已证明 frozen deletion 无法直接删除剩余 candidates。

必须记录起点：

```text
checkpoint SHA256
alive candidate list
candidate a,b
train/val assignment
train/val family-candidate cost matrix
structured heldout metrics
```

禁止重新训练 S4 seed1 来挑更好的 checkpoint。

---

# 3. 不允许使用 GT 的地方

以下步骤全部 learner-visible：

```text
specialist detection
parent proposal
absorption pair ranking
adaptation
commit / rollback
stop decision
```

禁止使用：

```text
GT mechanism id
GT purity
M_real = 3
GT parent pair
GT v
```

GT 只允许在实验结束后做 evaluator-only diagnosis。

---

# 4. 三个模型假设

对于 candidate \(j\)：

### H_keep

\[
H_{\rm keep}:
\text{candidate }j\text{ 是一个独立且必要的 mechanism}
\]

### H_drop-frozen

\[
H_{\rm frozen}:
\text{直接删除 }j,\text{其余 mechanism 不适应}
\]

这就是 ADP13 当前 existence test。

### H_absorb

\[
H_{\rm absorb}(j\rightarrow k):
\text{删除 }j,\text{由 candidate }k\text{ 在受约束条件下吸收其 contexts}
\]

ADP14 真正比较：

\[
\boxed{
H_{\rm keep}
\quad vs\quad
H_{\rm absorb}
}
\]

---

# 5. 自动发现 specialist → parent 候选

设 candidate \(j\) 当前 hard-assigned family 集合为：

\[
\mathcal F_j.
\]

对每个 \(k\neq j\)，定义 best-alternative concentration：

\[
P_{j\rightarrow k}
=
\frac{
\left|
\left\{
U\in\mathcal F_j:
k=
\arg\min_{\ell\neq j}E(U,\ell)
\right\}
\right|
}{
|\mathcal F_j|
}.
\]

分别在 train 与 validation family 上计算：

\[
P_{j\rightarrow k}^{train},
\qquad
P_{j\rightarrow k}^{val}.
\]

第一版 proposal gate：

```text
P_train >= 0.90
P_val   >= 0.80
```

如果某 candidate 在 validation 中 family 数少于 5：

```text
只要求 P_train >= 0.90
并把 low-val-support 标记进 telemetry
```

这只是 **pair proposal**，不是删除条件。

---

# 6. Replacement Gap

对 proposal pair \(j\rightarrow k\)：

\[
G_{j\rightarrow k}
=
\operatorname{median}_{U\in\mathcal F_j}
\left[
E(U,k)-E(U,j)
\right].
\]

同时记录：

```text
mean gap
median gap
p90 gap
max gap
fraction gap < 0.01
fraction gap < 0.02
```

不使用 GT。

---

# 7. 方向性检查

一个真正的 specialist → parent 关系应具有一定非对称性。

计算反向：

\[
P_{k\rightarrow j},
\qquad
G_{k\rightarrow j}.
\]

定义：

\[
D_{j,k}
=
G_{k\rightarrow j}
-
G_{j\rightarrow k}.
\]

pair 排序采用：

1. 先按 \(P_{j\rightarrow k}^{val}\) 降序；
2. 再按 \(P_{j\rightarrow k}^{train}\) 降序；
3. 再按 \(G_{j\rightarrow k}^{val}\) 升序；
4. 最后按 \(D_{j,k}\) 降序。

不设置 GT-informed pair。

---

# 8. 每次只测试一个 Pair

每个 absorption cycle：

```text
1. 扫描全部 alive directional pairs
2. 生成 passing proposals
3. 选 rank #1 pair
4. 执行 absorption trial
5. trial PASS -> commit
6. trial FAIL -> rollback，并测试 rank #2
```

一个 cycle 最多测试：

```text
3 个 proposal pairs
```

如果前三个都失败：

```text
STOP current cycle
```

---

# 9. A0：Frozen Delete Baseline

对选中的 \(j\rightarrow k\) 先运行旧 baseline：

```text
delete j
freeze all remaining candidates
reassign
evaluate
```

保存：

\[
\Delta^{frozen}.
\]

A0 不允许 commit。

它只用于验证：

\[
\boxed{
\text{candidate 当前确实不能被简单拔掉}
}
\]

---

# 10. A1：Constrained Specialist Absorption（主实验）

对 proposal：

\[
j\rightarrow k
\]

执行：

### Step 1：Trial-copy

复制当前完整 checkpoint：

```text
trial checkpoint
```

原模型不被覆盖。

### Step 2：删除 child \(j\)

在 trial model 中：

```text
alive = alive - {j}
```

### Step 3：冻结其他 candidates

除 parent \(k\) 外：

\[
\boxed{\text{all other mechanisms frozen}}
\]

只允许更新：

```text
M_k
a_k
b_k
```

### Step 4：构建训练集合

设：

- \(\mathcal F_k^{old}\)：parent 原先负责的 families；
- \(\mathcal F_j^{child}\)：child 原先负责的 families。

吸收训练使用：

\[
\mathcal F_{absorb}
=
\mathcal F_k^{old}
\cup
\mathcal F_j^{child}.
\]

只使用 train families。

validation / structured heldout 不参与梯度更新。

---

# 11. A1 Loss

定义：

\[
L_{A1}
=
L_{\rm child}
+
\lambda_{\rm replay}L_{\rm parent}
+
\lambda_{\rm anchor}L_{\rm func}.
\]

固定：

\[
\lambda_{\rm replay}=1,
\qquad
\lambda_{\rm anchor}=1.
\]

不做 sweep。

---

# 12. Child Assimilation Loss

\[
L_{\rm child}
=
\frac1{|\mathcal F_j^{child}|}
\sum_{U\in\mathcal F_j^{child}}
E(U,k).
\]

目标：

> parent 学会 child 当前占据的 contexts。

---

# 13. Parent Replay Loss

\[
L_{\rm parent}
=
\frac1{|\mathcal F_k^{old}|}
\sum_{U\in\mathcal F_k^{old}}
E(U,k).
\]

目的：

\[
\boxed{
\text{吸收新 contexts 时不能忘掉 parent 已经掌握的 contexts}
}
\]

每个 minibatch 使用：

```text
50% child families
50% old-parent families
```

若某侧不足则循环采样。

---

# 14. Functional Anchor

在 absorption 前保存 parent：

\[
\mathcal M_k^{old}.
\]

构造 anchor pool：

```text
parent old train families
+
parent old validation states/actions
```

不使用 GT。

anchor loss：

\[
L_{\rm func}
=
\mathbb E_{(S,\alpha)\sim A_k}
\left[
\left\|
\mathcal M_k^{new}
\left(
S,a_k^{new}\tilde\alpha+b_k^{new}
\right)
-
\mathcal M_k^{old}
\left(
S,a_k^{old}\tilde\alpha+b_k^{old}
\right)
\right\|^2
\right].
\]

这不是要求参数不变，而是要求 parent 的已有功能在原 contexts 上保持。

---

# 15. Adaptation Budget

固定：

```text
absorption epochs = 20
batch size = 32
mechanism LR = 3e-4
coordinate LR = 1e-3
weight decay = same as S4
optimizer = same as S4
```

等价于：

```text
5 mini-rounds × 4 epochs
```

禁止“训练到通过为止”。

每 4 epochs 只保存 telemetry，不提前 commit。

---

# 16. Trial 后的重新 Assignment

20 epochs 后：

1. 在所有 train/val families 上重新计算：
   \[
   E(U,j)
   \]
2. top-1 reassignment；
3. 重新计算：
   ```text
   family counts
   assignment change
   candidate cost matrix
   ```
4. 再运行 structured predictive evaluation。

不能只测试 child 的原 families。

---

# 17. 主要模型选择指标

比较：

\[
M_{\rm keep}
\]

与：

\[
M_{\rm absorb}.
\]

对：

\[
d\in
\{
IID,\,
state,\,
realization,\,
base
\}
\]

定义：

\[
\Delta^{adapt}_d
=
NRMSE_d(M_{\rm absorb})
-
NRMSE_d(M_{\rm keep}).
\]

---

# 18. 数值稳定的 Bootstrap

ADP13 existence JSON 中部分 per-base ratio 在 target energy 接近 0 时出现极大数值。

ADP14 禁止：

```text
先算每个 base 的 NRMSE ratio
再直接平均这些 ratio
```

改为对 base ID bootstrap。

每次 bootstrap：

1. 重采样 base IDs；
2. 拼接被采样 bases 的全部 observations；
3. 计算：
   \[
   NRMSE
   =
   \sqrt{
   \frac{\sum\|e\|^2}
   {\sum\|y\|^2+\epsilon}
   }
   \]
4. 再计算 keep / absorb 差值。

运行：

```text
500 bootstrap replicates
```

保存：

```text
mean delta
median delta
95% CI
upper 95% CI
```

---

# 19. A1 Commit Gate

absorption trial 只有全部满足才允许 commit：

### Global domains

```text
IID:
upper95(ΔNRMSE) < 0.01

cross-state:
upper95(ΔNRMSE) < 0.01

cross-realization:
upper95(ΔNRMSE) < 0.01

cross-base:
upper95(ΔNRMSE) < 0.01
```

### Subgroups

所有预注册 subgroup：

```text
max ΔNRMSE < 0.02
```

### Parent-retention

在 parent 原 validation families：

```text
ΔNRMSE_parent_old < 0.01
```

### Child assimilation

在 child 原 validation families：

```text
NRMSE_absorb <= NRMSE_keep + 0.01
```

### Finite / safety

```text
all predictions finite
no NaN/Inf
```

GT purity / GT functional \(R^2\) 不进入 commit gate。

---

# 20. Commit 后动作

若 A1 PASS：

```text
hard commit delete child j
save absorbed parent k
recompute all assignments
freeze all mechanisms for 1 audit pass
run structured validation again
```

若 audit 仍 PASS：

```text
accept cycle
```

否则：

```text
rollback to pre-cycle checkpoint
mark pair failed
```

---

# 21. 下一轮 Absorption

成功一次后：

```text
重新扫描全部 directional pairs
重新计算 P、G、D
重新 rank
```

不能提前指定第二个 pair。

因此自动形成：

\[
M=5\rightarrow4\rightarrow\cdots
\]

停止条件不是 \(M=3\)，而是：

\[
\boxed{\text{没有 proposal pair 能通过 constrained absorption gate}}
\]

---

# 22. A2：No Functional Anchor Control

与 A1 完全相同，但：

\[
\lambda_{\rm anchor}=0.
\]

仍保留：

\[
L_{\rm child}
+
L_{\rm parent}.
\]

目的：

> 判断 functional anchor 是否真的防止 parent 在吸收 specialist 时发生 cross-context drift。

禁止删除 replay；否则一次改变两个因素。

---

# 23. A3：Wrong-Parent Negative Control

对 A1 自动 proposal 的 child \(j\)：

选择 learner-visible replacement 明显更差的 wrong parent：

\[
k_{\rm wrong}
\]

规则：

```text
在非 proposal candidates 中
选择 validation replacement gap 最大者
```

给它完全相同：

```text
20 epochs
same replay
same anchor
same LR
same evaluation
```

预期：

\[
\boxed{\text{A3 FAIL}}
\]

如果 wrong parent 也频繁 PASS，说明当前 absorbability criterion 太弱，网络只是重新拟合数据。

---

# 24. A4：Frozen Delete

就是 A0，但作为正式 negative baseline 保存完整结果：

\[
\boxed{
\text{frozen delete 应 FAIL，而 constrained absorb 可 PASS}
}
\]

这能区分：

```text
“现在有用”
```

和：

```text
“世界真的需要”
```

---

# 25. Stage A：Seed1 Proof-of-Mechanism

第一阶段只跑：

```text
ADP13 S4 seed1
```

变体：

```text
A1 constrained absorption
A2 no-anchor
A3 wrong-parent
A4 frozen-delete
```

主实验 A1。

---

# 26. Stage A PASS 条件

A1 必须：

```text
至少成功 commit 1 次 absorption
没有 GT-informed decision
没有 rollback 后仍强行删除
final structured gate PASS
```

最强 PASS：

```text
自动从 M=5 收缩到 M=3
并进入 B1
B1 PASS
```

若：

```text
0 次 absorption commit
```

则 ADP14 主假设不支持，STOP。

---

# 27. Final Z_D Gate

停止 absorption 后：

```text
family Hungarian accuracy > 0.90
fragmentation < 0.10
assignment change < 0.10

IID NRMSE < 0.035
cross-state NRMSE < 0.05
cross-realization NRMSE < 0.05
cross-base NRMSE < 0.05
```

GT evaluator-only：

```text
each GT mechanism best functional R² > 0.95
major candidate purity > 0.90
```

最后才报告：

```text
M_discovered
```

learner 不知道 \(M_{real}=3\)。

---

# 28. B1

只有 Final Z_D Gate PASS 才进入 B1。

保持 ADP13：

```text
inner steps = 60
restarts = 3
chunk size = 1024
IID = 20k
combination-101 = 20k
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

# 29. Stage B：三 Seed 安全检查

只有 seed1 A1 最强 PASS 后执行。

使用原 ADP13-S4：

```text
seed0
seed1
seed2
```

规则：

- seed1：重复完整 ADP14；
- seed0/2：从其最终模型运行 absorption scanner；
- 若 seed0/2 已经 \(M=3\)，预期 scanner 不应继续错误压缩真实机制。

Stage B 安全要求：

```text
seed0/2 false absorption = 0
seed1 rescue = PASS
```

如果 seed0/2 被继续压成 M<3：

\[
\boxed{\text{ADP14 model-selection rule FAIL}}
\]

---

# 30. Telemetry

每个 cycle 必须保存：

```text
alive candidates

train assignment
val assignment

E_train(U,j)
E_val(U,j)

P_train(j->k)
P_val(j->k)

replacement gap:
mean
median
p90
max

directionality D(j,k)

selected pair
candidate rank table

pre-absorption metrics
per-epoch absorption loss

child loss
parent replay loss
functional anchor loss

post-absorption assignment

IID/state/v/base metrics
bootstrap CI
subgroup metrics

commit / rollback
```

---

# 31. 必须生成的图

## Figure A：Directional replacement graph

节点：

```text
candidate
```

边：

\[
j\rightarrow k
\]

边宽：

\[
P_{j\rightarrow k}
\]

用于观察：

```text
core + specialist
```

结构是否自然出现。

## Figure B：Absorption learning curve

横轴：

```text
epoch
```

画：

```text
child NRMSE
parent-old NRMSE
anchor loss
global validation NRMSE
```

## Figure C：Before vs After Context Coverage

显示：

```text
pre-absorption candidate assignment
post-absorption assignment
```

## Figure D：Frozen vs Adapted Drop

比较：

\[
\Delta^{frozen}
\quad vs\quad
\Delta^{adapt}.
\]

这是 ADP14 最核心的图。

---

# 32. 失败分类

### F1：No Parent Proposal

没有任何 directional pair 满足 proposal gate。

说明当前 fragmentation 不是简单 specialist-parent 结构。

### F2：Frozen-Only Necessity

pair 看起来像 specialist-parent，但 adaptation 仍无法吸收。

### F3：Parent Drift

child contexts 被学会，但 parent 原 contexts 明显退化。

### F4：Memorization / Wrong-Parent Pass

正确 parent 与 wrong parent 都能通过。

说明网络容量太大，当前 test 不足以识别 mechanism identity。

### F5：Overcompression

在 seed0/2 或 seed1 中把必要 mechanism 继续删除。

### F6：Count Recovered, B1 Failed

family-level model selection正确，但 full-TP composition失败。

---

# 33. 输出目录

```text
outputs/
└── e0_adp14_specialist_absorption/
    ├── seed_001/
    │   ├── baseline/
    │   ├── pair_scan/
    │   ├── cycle_00/
    │   │   ├── A1_constrained/
    │   │   ├── A2_no_anchor/
    │   │   ├── A3_wrong_parent/
    │   │   └── A4_frozen_delete/
    │   ├── cycle_01/
    │   ├── final/
    │   └── final_verdict.json
    ├── stage_b_seed0/
    ├── stage_b_seed2/
    └── aggregate/
```

---

# 34. 每次 Cycle Summary

```json
{
  "cycle": 0,
  "alive_before": 5,

  "child": "C?",
  "parent": "C?",

  "proposal": {
    "P_train": 1.0,
    "P_val": 1.0,
    "median_gap": 0.0,
    "directionality": 0.0
  },

  "frozen_delete_pass": false,

  "A1": {
    "absorption_pass": true,
    "commit": true,
    "rollback": false
  },

  "alive_after": 4
}
```

---

# 35. 命令模板

```bash
# Smoke test
python run_adp14.py --stage smoke

# 查看 seed1 pair scan，不训练
python run_adp14.py --stage scan --seed 1

# seed1 主实验
python run_adp14.py --stage proof --seed 1 --variant A1

# controls
python run_adp14.py --stage proof --seed 1 --variant A2
python run_adp14.py --stage proof --seed 1 --variant A3
python run_adp14.py --stage proof --seed 1 --variant A4

# Stage A 汇总
python analyze_adp14.py --stage proof --seed 1

# 只有 Stage A 最强 PASS 后
python run_adp14.py --stage safety_3seed --seeds 0,1,2

# 最终分析
python analyze_adp14.py --stage all
```

---

# 36. Stop Rule

### STOP-A

如果 seed1：

```text
没有任何 A1 absorption commit
```

停止。

不调 threshold，不延长 epoch。

### STOP-B

如果 A3 wrong-parent 也 PASS：

```text
STOP
```

说明 absorbability test 没有 mechanism discrimination。

### STOP-C

如果 Stage B seed0/2 出现 false absorption：

```text
STOP
```

说明会过压缩真实机制。

---

# 37. 结果解释

## Case A：A1 把 seed1 自动 5→3，B1 PASS；A3 FAIL

最强支持：

\[
\boxed{
\text{额外 candidates 是 optimization-created context specialists，
不是独立 }Z_D
}
\]

说明下一步 ADP15 应直接研究：

\[
\boxed{\text{cross-context predictive identity formation}}
\]

目标从“事后吸收 specialist”前移到“训练时避免 specialist 形成”。

---

## Case B：A1 能删 1 个，但不能完全恢复

支持部分 fragmentation hypothesis。

下一步应分析剩余 candidate 是：

```text
更深 structural distortion
还是另一个 parent relation 没被 proposal metric 捕获
```

---

## Case C：A1 完全失败

说明 frozen necessity 不只是训练路径造成的轻度 specialization。

应回到 identity formation，研究 early hard assignment basin locking。

---

## Case D：A3 也 PASS

说明当前网络可以在有限 adaptation 内任意重拟合。

这时不能把 absorption 当作 mechanism identity evidence。

下一步必须引入更强：

\[
\boxed{\text{out-of-context / extrapolative evaluation}}
\]

---

# 38. ADP14 的理论意义

ADP13 的 frozen-delete test 实际问：

\[
\boxed{
\text{candidate 在当前已经形成的参数化模型中有没有用？}
}
\]

ADP14 更接近 model comparison：

\[
\boxed{
\text{在删除 candidate 后，较小模型是否能通过有限受约束重优化恢复同样的预测能力？}
}
\]

因此它区分：

\[
\text{current utility}
\]

与：

\[
\text{structural necessity}.
\]

如果 absorption 成功：

\[
\boxed{
\text{一个 candidate 可以“现在有用”，但仍不是一个独立 }Z_D.
}
\]

---

# 39. 一句话执行标准

\[
\boxed{
\text{先用 learner-visible replacement evidence 自动找 specialist→parent，
再删除 specialist，只允许 parent 在 replay + functional anchor 保护下有限适应；
如果较小模型恢复跨 context 预测能力，则把该 specialist 判为 optimization-created fragmentation。}
\]
