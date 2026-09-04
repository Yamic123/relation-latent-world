# ADP9 实验执行方案：基于功能等价的逐步删冗余机制

## 0. 实验定位

ADP8 已证明：duplicate identity 成立，ADP7 的失败来自 merge 后 joint fine-tuning 的 catastrophic drift。

因此 ADP9 不再重新训练 merged mechanism，而采用：

\[
\boxed{
\text{识别功能等价副本}
\rightarrow
\text{保留一个代表}
\rightarrow
\text{删除另一个}
\rightarrow
\text{重新 assignment}
}
\]

并重复，直到：

\[
\boxed{
\text{剩余 candidate 中不存在可安全删除的 duplicate。}
}
\]

注意：\(M_{\rm real}=3\) 只能由 evaluator 使用，不能作为 learner 的停止条件。

---

## 1. 核心假设

若两个 candidate \(C_j,C_k\) 满足：

1. 跨 state 功能等价；
2. 跨 realization 功能等价；
3. 跨 base / TP 功能等价；
4. 任一副本均可直接替代另一个；
5. 替代时无需更新 mechanism 参数；

则：

\[
\boxed{
C_j,C_k
\text{ 是同一个 }Z_D\text{ 的冗余副本。}
}
\]

因此全局 mechanism count 可以通过逐步删除 functional duplicate 自然收缩。

---

## 2. 约束

禁止：

```text
使用 M_real 作为停止条件
按 usage 单独删除
按参数距离单独删除
按 embedding 距离单独删除
重新训练 mechanism
把两个网络拼接成新网络
```

允许：

```text
删除 duplicate
重新做 family assignment
必要时只微调 surviving candidate 的 a,b
```

默认：

\[
\boxed{\mathcal M_j\text{ 全程冻结}}
\]

---

## 3. 起点

使用 ADP6-A2 / ADP8 已确认 duplicate identity 的 checkpoint。

初始：

```text
Mmax = 5
alive = {C1,C2,C3,C4,C5}
```

ADP9 代码不能硬编码 GT pair。

duplicate pair 必须由 learner-visible functional equivalence 自动发现。

---

## 4. 总体流程

```text
ADP9-0  构建 candidate 功能等价图
   ↓
A1      选择最高置信 duplicate pair
   ↓
A2      选择 survivor 并 hard prune
   ↓
A3      全量 family reassignment
   ↓
A4      删除后无损性检查
   ↓
若仍存在 duplicate
   └── 回到 A1
否则
   ↓
A5      自动停止，得到 M_discovered
   ↓
B       回到完整 TP participation
```

任一单步 prune FAIL：rollback。

---

## 5. ADP9-0：功能等价图

对所有 alive candidates 两两计算：

```text
cross-state prediction R²
cross-realization prediction R²
cross-base prediction R²
direct-replacement NRMSE increase
family response disagreement
|a_j-a_k|
|b_j-b_k|
```

GT 只用于 evaluator。

### duplicate edge 判据

pair \((j,k)\) 只有同时满足：

```text
cross-state prediction R² > 0.995
cross-realization prediction R² > 0.995
cross-base prediction R² > 0.995
direct replacement max NRMSE increase < 0.01
```

才建立 duplicate edge。

构图：

```text
node = alive candidate
edge = verified duplicate
edge weight = equivalence confidence
```

---

## 6. A1：每轮只选一个 pair

若 duplicate edge 非空：

\[
(j^*,k^*)
=
\arg\max score_{jk}
\]

每轮只处理一个 pair，不一次删多个。

这样可以观察每次删除后 assignment landscape 是否变化。

---

## 7. survivor 选择

不能随机。

定义 learner-visible quality：

\[
Q_j
=
0.5L_j^{heldout}
+
0.2U_j
+
0.3V_j
\]

其中：

- \(L_j^{heldout}\)：heldout prediction loss；
- \(U_j\)：assignment instability；
- \(V_j\)：state / realization coverage penalty。

选择：

\[
C_{\rm keep}
=
\arg\min Q
\]

另一个记为：

\[
C_{\rm drop}
\]

---

## 8. A2：hard prune

执行：

```text
freeze all mechanisms
remove C_drop
keep C_keep unchanged
```

禁止：

```text
mechanism averaging
parameter interpolation
mechanism fine-tuning
new merged network
```

先直接删除，不动：

\[
a_{\rm keep},b_{\rm keep}
\]

只有若 functional identity 不变，但存在轻微 coordinate mismatch，才允许冻结 \(\mathcal M_{\rm keep}\) 后只微调 \(a,b\)。

---

## 9. A3：删除后重新 assignment

对 surviving candidate 重新计算：

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
a_j\alpha_{U,r}+b_j
\right)
\right)
\]

然后：

\[
c_U=\arg\min_{j\in alive}E(U,j)
\]

此阶段所有 \(\mathcal M_j\) 保持冻结。

---

## 10. A4：单次删除 gate

每删一个 candidate，必须重新评估：

```text
IID heldout
cross-state
cross-realization
cross-base
family assignment
functional R²
fragmentation
assignment stability
```

### 单步 PASS

```text
max heldout NRMSE increase < 0.01

三个 evaluator GT family:
best functional R² 均 > 0.95

family Hungarian accuracy 下降 < 0.03
mean fragmentation 上升 < 0.03
3 次重复 reassignment 后 assignment change < 0.10
```

若 PASS：

```text
commit deletion
```

若 FAIL：

```text
rollback
blacklist this pair at current round
test next duplicate pair
```

---

## 11. 逐步 prune 伪代码

```python
alive = all_candidates
blacklist = set()

while True:
    graph = build_functional_equivalence_graph(alive)

    pairs = [
        e for e in graph.valid_duplicate_edges()
        if e not in blacklist
    ]

    if len(pairs) == 0:
        break

    pair = highest_confidence_pair(pairs)
    keep, drop = choose_survivor(pair)

    snapshot = save_state()

    delete(drop)
    reassign_all_families()

    if prune_passes_all_gates():
        commit()
        blacklist.clear()
    else:
        rollback(snapshot)
        blacklist.add(pair)
```

---

## 12. 停止条件

禁止：

```python
if alive_count == 3:
    stop
```

正确停止条件：

\[
\boxed{
\mathcal P_{\rm dup}=\emptyset
}
\]

即剩余 candidate 中不存在任何能通过 functional equivalence + direct replacement gate 的 pair。

此时：

\[
M_{\rm discovered}=|alive|
\]

训练结束后 evaluator 才比较：

\[
M_{\rm discovered}
\]

与：

\[
M_{\rm real}=3
\]

---

## 13. 预期但不可硬编码的过程

若当前理解正确，可能出现：

```text
start: 5 candidates

round 1:
delete one of C2/C5
alive = 4

round 2:
delete one of C1/C4
alive = 3

recompute graph:
no valid duplicate edge
STOP
```

这只是实验预期，不能写入 learner 规则。

---

## 14. Negative controls

### NC-1：随机删除

每轮随机删一个 alive candidate。

预期 functional R² 明显下降。

### NC-2：按 usage 删除

删 usage 最低 candidate。

验证：

\[
\boxed{
\text{low usage}\neq\text{duplicate identity}
}
\]

### NC-3：错误 pair 强删

对 equivalence gate 不通过的 pair 强行删除一个。

预期：

```text
functional R² 下降
heldout NRMSE 上升
family mixing 增加
```

### NC-4：删除后 joint fine-tuning

作为 ADP7/ADP8 已知错误 baseline。

预期 cross-realization drift。

### NC-5：一次删多个

若当前 graph 有多条 duplicate edge，一次同时删多个。

与逐步 prune 比较，验证“逐步删除 + 每步 reassignment”是否更稳定。

---

## 15. 单轮 telemetry

必须保存：

```text
prune_round
alive_candidates
candidate_count

duplicate_edges
edge_scores

selected_pair
survivor
dropped_candidate

pre-prune metrics
post-prune metrics

IID NRMSE
cross-state NRMSE
cross-realization NRMSE
cross-base NRMSE

functional R² matrix
matched functional R²

family Hungarian accuracy
mean fragmentation
assignment change

candidate usage
candidate family count

coordinate a,b

commit / rollback
```

---

## 16. A5 最终 PASS

development seed 0：

```text
stop reason = no valid duplicate edge

M_discovered = 3        # evaluator-only check

三个 matched functional R² > 0.95

IID NRMSE < 0.03
cross-state NRMSE < 0.05
cross-realization NRMSE < 0.05
cross-base NRMSE < 0.05

family Hungarian accuracy > 0.90
mean fragmentation < 0.10
assignment change < 0.10
```

---

## 17. Multi-seed

seed 0 PASS 后：

```text
seeds = 0,1,2
```

要求：

```text
至少 2/3:
M_discovered = 3
且所有 functional gates 通过
```

正式：

```text
seeds = 0..9
```

Strong pass：

\[
\boxed{8/10}
\]

恢复相同 global count 且保持 functional identity。

---

## 18. B：回到完整 TP participation

只有 ADP9 pruning 成功后执行。

此时：

\[
Z_D
=
\{\mathcal M_1,\ldots,\mathcal M_{M_{\rm discovered}}\}
\]

已经完成：

```text
family identity
realization coordinate
duplicate removal
global count discovery
```

然后对 full TP：

\[
(S_i,p_i,\Delta S_i)
\]

推断：

\[
m_i^j,\alpha_i^j
\]

其中：

\[
v_i^j=a_j\alpha_i^j+b_j
\]

---

## 19. B1 support inference

枚举：

\[
2^{M_{\rm discovered}}
\]

个 support。

目标：

\[
\min_{m,\alpha}
D_S
\left(
\Delta S_i,
\sum_j
m_i^j
\mathcal M_j
\left(
S_i,
a_j\alpha_i^j+b_j
\right)
\right)
+
\lambda_P\|m_i\|_0
\]

### B1 PASS

```text
IID NRMSE < 0.05
101 NRMSE < 0.10
participation F1 > 0.90
min instance-effect R² > 0.90
min functional R² > 0.90
inactive Z2 FPR on 101 < 0.10
```

---

## 20. 输出目录

```text
outputs/
└── e0_adp9/
    ├── adp9_0_equivalence_graph/
    ├── prune_round_00/
    │   ├── graph.json
    │   ├── prune_decision.json
    │   ├── pre_metrics.json
    │   ├── post_metrics.json
    │   └── assignments.npz
    ├── prune_round_01/
    ├── final_pruned_model/
    ├── negatives/
    │   ├── random_delete/
    │   ├── usage_delete/
    │   ├── wrong_pair/
    │   ├── joint_finetune/
    │   └── batch_delete/
    └── b1_full_tp/
```

---

## 21. 推荐命令模板

```bash
python run_adp9.py --stage adp9_0 --seed 0
python run_adp9.py --stage adp9_prune --seed 0
python run_adp9.py --stage adp9_verify --seed 0

python run_adp9.py --stage adp9_nc_random_delete --seed 0
python run_adp9.py --stage adp9_nc_usage_delete --seed 0
python run_adp9.py --stage adp9_nc_wrong_pair --seed 0
python run_adp9.py --stage adp9_nc_joint_finetune --seed 0
python run_adp9.py --stage adp9_nc_batch_delete --seed 0

python run_adp9.py --stage adp9_b1 --seed 0
```

---

## 22. 最重要结果表

| Round | Alive before | Duplicate pair | Survivor | Alive after | Functional R² | Fragmentation | Stable? | Commit? |
|---|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 5 |  |  |  |  |  |  |  |
| 1 | 4 |  |  |  |  |  |  |  |
| stop |  | no duplicate | — |  |  |  |  | — |

对照表：

| Method | Final M | Functional R² | Cross-v | Fragmentation | Stable |
|---|---:|---:|---:|---:|---:|
| ADP9 gradual prune |  |  |  |  |  |
| Random delete |  |  |  |  |  |
| Usage delete |  |  |  |  |  |
| Batch delete |  |  |  |  |  |
| Joint fine-tune |  |  |  |  |  |

---

## 23. 失败解释

### 删 duplicate 后 functional 下降

pair 只是局部相似，不是全局可替代。rollback。

### functional 不变但 fragmentation 上升

被删副本可能承担 assignment stabilization 作用。检查 survivor selection。

### 删到 4 个后无 duplicate edge

learner 必须停在：

\[
M_{\rm discovered}=4
\]

不能因为 evaluator 知道 \(M_{\rm real}=3\) 而继续强删。

### 删到 2 个仍存在 duplicate edge

equivalence criterion 太宽松，发生 false merge。

### joint fine-tuning 才失败

与 ADP8 一致，说明 pruning 正确，但 mechanism 更新破坏了已形成 identity。

---

## 24. 科学意义

ADP9 尝试验证：

\[
\boxed{
\text{全局 mechanism count 能否通过反复删除功能等价副本自然出现。}
}
\]

因此：

\[
M_{\rm discovered}
\]

不是预设目标，而是：

\[
\boxed{
\text{当剩余 candidate 之间已经不存在可安全删除的 functional duplicate 时的模型复杂度。}
}
\]

若成功，当前 E0 链条可更新为：

\[
\boxed{
\text{response family}
\rightarrow
\text{family identity}
\rightarrow
\text{统一 realization coordinate}
\rightarrow
\text{functional duplicate pruning}
\rightarrow
Z_D
}
\]

---

## 25. 一句话执行标准

\[
\boxed{
\text{每次只删一个经 structured functional equivalence 验证的 duplicate；
删除后冻结 mechanism、重新 assignment、重新验证；
直到不存在任何可安全删除的 pair。}
}
