# ADP15 实验执行方案：Capacity vs Shared-Representation Diagnostic

## 0. 实验目标

ADP14 已经确认两点：

1. fragmentation 主要沿 **state context** 发生，而不是 action sign / magnitude / realization；
2. constrained absorption 能改善 child context，但同时损害 parent 原 context，表现为明显的 **cross-state interference**。

因此 ADP15 不再研究 pruning，也不再调 absorption threshold，而是直接区分：

\[
\boxed{
\text{A. parent-specific capacity 不足}
}
\]

与：

\[
\boxed{
\text{B. shared trunk representation 本身已经 context-fragmented}
}
\]

核心问题：

> 同一个 candidate 如果拥有更多独立表达能力，是否能同时覆盖 parent / child 两个 state regions？如果不能，是否必须修改 shared trunk 才能统一它们？

---

# 1. 实验对象

主 pair：

\[
\boxed{C5\rightarrow C4}
\]

原因：

- learner-visible proposal 最强；
- ADP14 中 adaptation 相比 frozen delete 改善最明显；
- 但仍同时出现 child residual + parent drift。

辅助 pair：

\[
\boxed{C2\rightarrow C3}
\]

用于检验结论是否只对一个 pair 成立。

主实验先只跑：

```text
ADP13-S4 seed1 round39 checkpoint
C5 -> C4
```

只有主 pair 结论明确后，再跑 C2→C3 复现。

---

# 2. 固定内容

保持：

```text
same source checkpoint
same train / val / heldout splits
same family assignments at start
same family coordinate alpha
same a,b initialization
same optimizer family
same batch size
same structured evaluation
same bootstrap method
```

禁止：

```text
使用 GT mechanism label 做任何训练/选择
使用 M_real=3 决定停止
放宽 final gate
延长训练直到通过
```

---

# 3. 变量：逐级开放参数自由度

定义 4 个 Level。

## L0：Frozen baseline

删除 child，不训练任何参数。

作用：复现 ADP14 frozen-delete baseline。

---

## L1：Adapter-only

只更新：

```text
parent adapter row
parent a
parent b
```

shared trunk 全冻结。

这就是 ADP14 A1。

作用：作为容量最低基线。

---

## L2：Expanded Parent-Specific Head

shared trunk 继续冻结，但给 parent 增加独立表达能力。

推荐结构：

```text
shared trunk output h
        |
        +--> old adapter path
        |
        +--> parent-only residual MLP
```

parent residual：

\[
r_k(h,v)
=
W_2\sigma(W_1[h,v])
\]

输出：

\[
\mathcal M_k^{new}
=
\mathcal M_k^{old}
+
r_k
\]

第一版固定：

```text
hidden dim = 64
1 hidden layer
activation = same as mechanism bank
residual output init = 0
```

只训练：

```text
parent adapter
parent residual MLP
parent a,b
```

shared trunk 仍冻结。

---

## L3：Protected Shared-Trunk Adaptation

允许 shared trunk 更新，但必须保护所有其他 candidates 的已有功能。

可训练：

```text
shared trunk
parent adapter/head
parent a,b
```

冻结：

```text
other candidate-specific adapters
other candidate-specific a,b
```

同时增加 global functional distillation：

\[
L_{\rm protect}
=
\sum_{\ell\neq k}
\mathbb E_{(S,\alpha)\sim A_\ell}
\left[
\|
\mathcal M_\ell^{new}(S,\alpha)
-
\mathcal M_\ell^{old}(S,\alpha)
\|^2
\right]
\]

目的：

\[
\boxed{
\text{允许 shared representation 改变，
但不允许其他机制功能漂移}
}
\]

---

# 4. 训练目标

三个 trainable level 共用：

\[
L
=
L_{\rm child}
+
L_{\rm parent}
+
\lambda_{\rm anchor}L_{\rm parent-anchor}
+
\lambda_{\rm protect}L_{\rm protect}
\]

其中：

### Child assimilation

\[
L_{\rm child}
=
\frac1{|\mathcal F_j|}
\sum_{U\in\mathcal F_j}
E(U,k)
\]

### Parent replay

\[
L_{\rm parent}
=
\frac1{|\mathcal F_k|}
\sum_{U\in\mathcal F_k}
E(U,k)
\]

### Parent anchor

\[
L_{\rm parent-anchor}
=
\|\mathcal M_k^{new}-\mathcal M_k^{old}\|^2
\]

在 parent-old anchor set 上计算。

### Other-candidate protection

L1/L2：

\[
\lambda_{\rm protect}=0
\]

L3：

\[
\lambda_{\rm protect}=1
\]

固定：

```text
lambda_anchor = 1
lambda_protect = 1 (L3 only)
```

不做 sweep。

---

# 5. Minibatch 组成

每个 training batch：

```text
50% child families
50% parent-old families
```

L3 额外独立采样：

```text
other-candidate anchor batch
```

用于 \(L_{\rm protect}\)。

---

# 6. 训练预算

统一：

```text
epochs = 20
batch size = 32
mechanism LR = 3e-4
coordinate LR = 1e-3
weight decay = same as ADP14
optimizer = same as ADP14
```

L3 shared-trunk LR：

```text
trunk LR = 1e-4
```

不能因为某个 level 没过而增加 epochs。

---

# 7. 每 4 Epoch 保存一次诊断

保存时点：

```text
epoch 0
epoch 4
epoch 8
epoch 12
epoch 16
epoch 20
```

保存：

```text
child train NRMSE
parent train NRMSE
child val NRMSE
parent val NRMSE
global val NRMSE

cross-state NRMSE
cross-v NRMSE
cross-base NRMSE

parent adapter norm
parent residual norm
trunk update norm

a,b
```

---

# 8. 核心 Gate

每个 level 结束后，对较小模型重新 assignment，并评估：

\[
\Delta_d
=
NRMSE_d(M_{reduced})
-
NRMSE_d(M_{keep})
\]

要求：

```text
IID upper95 bootstrap Δ < 0.01
state upper95 bootstrap Δ < 0.01
realization upper95 bootstrap Δ < 0.01
base upper95 bootstrap Δ < 0.01

max subgroup Δ < 0.02

parent-old val Δ < 0.01
child val NRMSE <= keep + 0.01

all predictions finite
```

只有全部满足，判该 level：

```text
ABSORB PASS
```

---

# 9. 关键比较

## Case 1：L1 FAIL，L2 PASS

结论：

\[
\boxed{
\text{主要问题是 parent-specific capacity 不足}
}
\]

shared trunk 基本够用，只是 adapter 太弱。

下一步：

```text
重设计 mechanism-specific head
```

而不是改 identity principle。

---

## Case 2：L1 FAIL，L2 FAIL，L3 PASS

结论：

\[
\boxed{
\text{shared trunk representation 本身限制了跨-state统一}
}
\]

即：

\[
\boxed{
\text{当前 shared representation 已经 context-fragmented}
}
\]

下一步应研究：

```text
cross-context representation learning
out-of-context predictive identity
shared trunk regularization / restructuring
```

---

## Case 3：L1/L2/L3 全 FAIL

说明：

\[
\boxed{
\text{当前 pair 不只是简单容量/共享表征问题}
}
\]

需要考虑：

```text
两个 candidates 学到了真正不同的 conditional functions
或
shared coordinate / family identity 仍有更深问题
```

这时才进入新的机制结构诊断。

---

## Case 4：L2 与 L3 都 PASS

比较：

```text
L2 参数量
L3 trunk drift
other candidate functional retention
```

优先选择 L2，因为它更局部、更符合稳定 \(Z_D\) 的需求。

---

# 10. 非常重要的 Negative Controls

## NC1：Wrong Parent

对 C5：

```text
使用 learner-visible worst replacement parent
```

分别跑 L2/L3。

预期：

```text
FAIL
```

如果 wrong parent 也 PASS：

\[
\boxed{
\text{模型只是有足够容量重拟合，不代表 mechanism identity 一致}
}
\]

---

## NC2：No Parent Replay

在 L2 主 pair 上移除 \(L_{\rm parent}\)。

预期：

```text
child improve
parent drift increase
```

用于确认 replay 的必要性。

---

## NC3：No Global Protection

在 L3 上：

```text
lambda_protect = 0
```

如果 child/parent 能吸收但其他 candidates 漂移：

\[
\boxed{
\text{说明 shared trunk 可修，但会破坏其他 }Z_D
}
\]

---

# 11. 3-Step 执行顺序

### Step A：主 pair C5→C4

顺序：

```text
L0
L1
L2
L3
```

先不跑 controls。

### Step B：根据主 pair 结果跑必要 controls

- 若 L2 PASS：跑 wrong-parent L2 + no-replay L2
- 若只有 L3 PASS：跑 wrong-parent L3 + no-protection L3

### Step C：辅助 pair C2→C3

只复现主 pair 中“最小成功 level”。

例如：

```text
C5→C4 在 L2 PASS
=> C2→C3 只跑 L2
```

若主 pair所有 level FAIL：

```text
STOP，不跑辅助 pair
```

---

# 12. 结果判定标准

## Strong evidence for capacity bottleneck

要求：

```text
C5→C4 L2 PASS
wrong-parent L2 FAIL
C2→C3 L2 至少结构和预测明显改善
```

支持：

\[
\boxed{
\text{ADP14 失败主要来自 adapter capacity 不足}
}
\]

---

## Strong evidence for representation bottleneck

要求：

```text
C5→C4 L2 FAIL
C5→C4 L3 PASS
wrong-parent L3 FAIL
no-protection L3 出现其他 mechanism drift
```

支持：

\[
\boxed{
\text{shared representation 是主要瓶颈}
}
\]

---

# 13. Telemetry

必须保存：

```text
per-epoch losses
per-level checkpoint
parameter update norms
parent/child assignment matrices
family-candidate costs
structured heldout metrics
bootstrap distributions
subgroup metrics

other-candidate function drift
```

L3 必须额外保存：

```text
shared trunk parameter norm
shared trunk delta norm
per-candidate output drift
```

---

# 14. 必须生成的图

### Figure A：Capacity Ladder

横轴：

```text
L0 L1 L2 L3
```

纵轴：

```text
global ΔNRMSE
parent-old Δ
child NRMSE
```

### Figure B：Child vs Parent Trade-off

横轴：

```text
child NRMSE
```

纵轴：

```text
parent-old NRMSE
```

每 4 epochs 一个点。

### Figure C：Other-Candidate Drift

仅 L3：

```text
C1/C2/C3/C5 function drift
```

### Figure D：State-Region Coverage

展示 parent/child 两个 state region 上：

```text
before
L2 after
L3 after
```

预测误差分布。

---

# 15. 输出目录

```text
outputs/
└── e0_adp15_capacity_representation/
    ├── C5_to_C4/
    │   ├── L0_frozen/
    │   ├── L1_adapter/
    │   ├── L2_parent_head/
    │   ├── L3_shared_trunk/
    │   ├── controls/
    │   └── verdict.json
    ├── C2_to_C3/
    └── aggregate/
```

---

# 16. 命令模板

```bash
# smoke
python run_adp15.py --stage smoke

# 主 pair 全部 level
python run_adp15.py --stage main --pair C5:C4

# 单独 level
python run_adp15.py --pair C5:C4 --level L2
python run_adp15.py --pair C5:C4 --level L3

# controls
python run_adp15.py --stage controls --pair C5:C4

# 辅助 pair
python run_adp15.py --stage replicate --pair C2:C3

# 汇总
python analyze_adp15.py
```

---

# 17. Stop Rule

### STOP-1

若：

```text
L1 FAIL
L2 FAIL
L3 FAIL
```

则停止，不调超参数。

### STOP-2

若 wrong-parent control PASS：

```text
STOP
```

说明当前 absorption test 不具 mechanism discrimination。

### STOP-3

若 L3 通过主 pair但造成其他 candidate 明显 drift：

```text
不能记作 PASS
```

必须先解决全局稳定性。

---

# 18. 本实验最终回答的问题

ADP15 不直接解决最终 \(Z_D\) discovery。

它只负责定位 ADP14 失败属于哪一种：

\[
\boxed{
\text{adapter capacity bottleneck}
}
\]

还是：

\[
\boxed{
\text{shared representation bottleneck}
}
\]

还是：

\[
\boxed{
\text{更深的 mechanism mismatch}
}
\]

只有定位完成后，下一步才决定：

- 改 mechanism-specific head；
- 改 shared representation；
- 或回到 identity formation 本身。

---

# 19. 一句话执行标准

\[
\boxed{
\text{对同一 specialist→parent pair 逐级开放参数自由度；
如果增加 parent-specific capacity 即可吸收，问题在 adapter；
如果必须改 shared trunk 才能吸收，问题在 shared representation；
如果都不行，再怀疑更深的 mechanism mismatch。}
\]
