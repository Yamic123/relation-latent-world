# ADP16 实验执行方案：C2/C3 Coordinate-Mismatch vs Conditional-Function Diagnostic

## 0. 实验目标

ADP15 已经得到一个关键分化：

- `C5→C4`：L1 adapter-only FAIL，L2 expanded parent-specific head PASS  
  \[
  \Rightarrow \text{主要是 parent-specific capacity bottleneck}
  \]
- `C2→C3`：同规格 L2 仍 FAIL，虽然 family accuracy / fragmentation 明显改善，但 predictive equivalence 没有恢复。

因此 ADP16 不再继续扩大一般容量，而只回答：

\[
\boxed{\text{C2/C3 为什么不能像 C5/C4 一样被统一？}}
\]

重点区分四种原因：

\[
\boxed{A.\ \text{realization-coordinate / gauge mismatch}}
\]

\[
\boxed{B.\ \text{warm-start / optimization path locking}}
\]

\[
\boxed{C.\ \text{shared representation bottleneck}}
\]

\[
\boxed{D.\ \text{更深的 state-conditional function mismatch / family identity 错配}}
\]

---

# 1. 实验对象

固定使用：

```text
ADP13-S4
seed = 1
round = 39
pair = C2 -> C3
```

起点 checkpoint 与 ADP15 完全相同。

不重新训练 S4，不挑 checkpoint。

---

# 2. 已知基线

复用 ADP15：

```text
C2 -> C3
L2 expanded parent-specific head
20 epochs
```

结果：

```text
max bootstrap upper95 = 0.04025
max subgroup Δ        = 0.02626
parent-old Δ          = 0.09602
child keep NRMSE      = 0.08128
child after NRMSE     = 0.09543
family accuracy       = 0.95596
fragmentation         = 0.04570
ABSORB                FAIL
```

因此 ADP16 不重复把“再加一个同规格 residual head”当主实验。

---

# 3. 总体实验结构

ADP16 分两阶段。

## Stage A：不训练或极少训练的机制诊断

回答：

> C2/C3 的差异能否主要由 realization coordinate 重新参数化解释？

## Stage B：Union Mechanism Intervention

回答：

> 如果不继承 C3 的已有 basin，而重新训练一个统一 mechanism，能否同时解释 C2/C3？

---

# 4. Stage A1：Shared-State Functional Disagreement Map

构造 learner-visible evaluation grid：

```text
states:
- C2 current train/val states
- C3 current train/val states
- existing cross-state heldout states

alpha:
- q05, q20, q35, q50, q65, q80, q95
```

分别计算：

\[
f_2(S,\alpha)=\mathcal M_2(S,a_2\alpha+b_2)
\]

\[
f_3(S,\alpha)=\mathcal M_3(S,a_3\alpha+b_3)
\]

定义：

\[
D_{raw}(S,\alpha)
=
\frac{
\|f_2(S,\alpha)-f_3(S,\alpha)\|
}{
\sqrt{\frac12(\|f_2\|^2+\|f_3\|^2)}+\epsilon
}
\]

按 state source 分组：

```text
C2-region
C3-region
cross-state
```

保存 mean / median / p90。

---

# 5. Stage A2：Coordinate-Rescue Test

核心问题：

> 是否存在一个 state-independent coordinate map，使 C3 的函数可以解释 C2？

定义 affine coordinate bridge：

\[
g(v)=\gamma v+\delta
\]

寻找：

\[
(\gamma^*,\delta^*)
=
\arg\min_{\gamma,\delta}
\mathbb E_{(S,\alpha)\in train}
\left[
\|
\mathcal M_3(S,g(v_2))
-
\mathcal M_2(S,v_2)
\|^2
\right]
\]

其中：

\[
v_2=a_2\alpha+b_2.
\]

训练：

```text
Adam
steps = 500
LR = 1e-2
仅训练 gamma, delta 两个标量
```

然后在：

```text
validation states
cross-state heldout
cross-realization heldout
cross-base heldout
```

评估。

---

# 6. Coordinate-Rescue PASS

定义：

\[
R^2_{bridge}
=
R^2
\left(
\mathcal M_3(S,g(v_2)),
\mathcal M_2(S,v_2)
\right)
\]

要求：

```text
validation R² > 0.995
cross-state R² > 0.995
cross-realization R² > 0.995
cross-base R² > 0.995
```

若 PASS：

\[
\boxed{\text{C2/C3 的主要差异可以由 state-independent realization gauge 解释}}
\]

则下一步优先修 shared realization coordinate。

---

# 7. Stage A3：State-Dependent Bridge Diagnostic

若 affine bridge FAIL，再允许：

\[
g(v,S)=\gamma(S)v+\delta(S)
\]

其中：

```text
gamma(S), delta(S)
由一个极小 state-MLP 输出
hidden dim = 16
```

只用于诊断，不能作为最终 \(Z_D\) 模型。

训练：

```text
steps = 1000
LR = 1e-3
shared trunk / C2 / C3 全冻结
```

如果：

```text
state-independent bridge FAIL
state-dependent bridge PASS
```

支持：

\[
\boxed{\text{差异是 state-conditioned，而不是单纯 realization gauge}}
\]

---

# 8. Stage B：Union Mechanism Test

核心思想：

> ADP15 的 L2 是“从 C3 出发吸收 C2”。  
> ADP16 要测试“不继承 C3 的已有 specialization，重新形成一个统一 mechanism”是否可行。

---

# 9. B1：Fresh Union Head + Single Shared Coordinate

新建 trial union candidate：

\[
C_U
\]

使用冻结的 shared trunk：

\[
h=Trunk(S)
\]

union head：

```text
input  = [h, v]
hidden = 64
1 hidden layer
GELU
output = 2
```

但：

```text
head 从新初始化
不继承 C3 adapter/head
```

使用一个统一 coordinate：

\[
v_U=a_U\alpha+b_U.
\]

初始化：

```text
a_U = mean(a2,a3)
b_U = mean(b2,b3)
head = standard random init
```

---

# 10. B1 训练数据

只使用：

\[
\mathcal F_2\cup\mathcal F_3
\]

train families。

batch：

```text
50% C2-source families
50% C3-source families
```

注意：`C2-source/C3-source` 只用于 batch balance，不作为模型输入。

---

# 11. B1 训练预算

固定：

```text
epochs = 40
batch size = 32
head LR = 3e-4
coordinate LR = 1e-3
weight decay = 1e-5
shared trunk frozen
```

---

# 12. B1 评估

分别评估：

```text
C2 old validation families
C3 old validation families
union validation
IID heldout
cross-state
cross-realization
cross-base
```

与原 keep-model 比较：

\[
\Delta_d
=
NRMSE_d(M_{union})
-
NRMSE_d(M_{keep})
\]

---

# 13. B1 PASS Gate

要求：

```text
C2-old val NRMSE <= keep + 0.01
C3-old val NRMSE <= keep + 0.01

IID upper95 bootstrap Δ < 0.01
cross-state upper95 bootstrap Δ < 0.01
cross-realization upper95 bootstrap Δ < 0.01
cross-base upper95 bootstrap Δ < 0.01

max subgroup Δ < 0.02
all finite
```

若 B1 PASS：

\[
\boxed{\text{单一 mechanism + 单一 coordinate 足以统一 C2/C3}}
\]

则 ADP15 失败主要是：

\[
\boxed{\text{warm-start / path locking}}
\]

---

# 14. B2：Fresh Union Head + Dual Coordinate（诊断）

若 B1 FAIL，运行 B2。

head 完全相同，但允许：

\[
v=
\begin{cases}
a_2'\alpha+b_2', & U\in\mathcal F_2\\
a_3'\alpha+b_3', & U\in\mathcal F_3
\end{cases}
\]

重要：

\[
\boxed{\text{source ID 只允许作为诊断 routing，不是最终模型允许的信息}}
\]

目的：

> 如果放松“必须共享一个 realization coordinate”，同一个 function head 能否覆盖两组 families？

---

# 15. B2 判定

若：

```text
B1 FAIL
B2 PASS
```

支持：

\[
\boxed{\text{主要瓶颈是 coordinate/gauge incompatibility}}
\]

若：

```text
B1 FAIL
B2 FAIL
```

进入 B3。

---

# 16. B3：Fresh Union Head + Protected Shared-Trunk Adaptation

仅当 B1/B2 均 FAIL 才运行。

允许：

```text
shared trunk
union head
union a,b
```

更新。

其他 candidates 全冻结。

增加 functional protection：

\[
L_{protect}
=
\sum_{\ell\notin\{2,3\}}
\mathbb E
\left[
\|
\mathcal M_\ell^{new}
-
\mathcal M_\ell^{old}
\|^2
\right]
\]

固定：

```text
trunk LR = 1e-4
head LR  = 3e-4
coordinate LR = 1e-3
lambda_protect = 1
epochs = 40
```

---

# 17. B3 PASS Gate

除 B1 gate 外，额外要求：

```text
other-candidate max functional drift < 0.01
```

若 B3 PASS：

\[
\boxed{\text{shared representation 是统一 C2/C3 所必需的瓶颈}}
\]

若 B3 也 FAIL：

\[
\boxed{\text{当前证据开始支持更深的 conditional-function / family-identity mismatch}}
\]

---

# 18. Wrong-Pair Control

为避免 fresh union head 只是因为容量足够而“什么都能拟合”，选择 learner-visible 明显不匹配的 pair：

```text
C2 + 一个非 C3 candidate
```

选择规则：

```text
基于 validation replacement gap
取与 C2 最差的 candidate
```

运行与 B1 完全相同的 fresh union single-coordinate 训练。

预期：

```text
FAIL
```

若 wrong-pair 也 PASS：

\[
\boxed{\text{B1 没有 mechanism discrimination，实验无效}}
\]

STOP。

---

# 19. Stage A/B 联合判定表

| A2 affine bridge | B1 single coord | B2 dual coord | B3 trunk | 结论 |
|---|---|---|---|---|
| PASS | 任意 | — | — | realization gauge mismatch 为主 |
| FAIL | PASS | — | — | warm-start / path locking |
| FAIL | FAIL | PASS | — | coordinate incompatibility |
| FAIL | FAIL | FAIL | PASS | shared representation bottleneck |
| FAIL | FAIL | FAIL | FAIL | deeper conditional-function / identity mismatch |

---

# 20. State-Conditional Diagnostic

如果最终进入最后一行：

```text
A2 FAIL
B1 FAIL
B2 FAIL
B3 FAIL
```

再做一个纯诊断。

按 learner-visible state features 将 union validation families 分成：

```text
state PCA / existing structured state buckets
```

计算每个 bucket：

```text
C2 error
C3 error
fresh-union error
C2-C3 disagreement
```

若错误呈明显 state-region 互补：

\[
\boxed{\text{支持真正的 state-conditional branch fragmentation}}
\]

---

# 21. 不允许的操作

正式实验开始后禁止：

```text
看到 B1 快通过就增加 epochs
调大 hidden dim
调 coordinate bridge 阈值
用 GT mechanism id 选择 pair
用 GT purity 做 commit
看到 B2 PASS 后把 dual coordinate 当最终方法
放宽 other-candidate drift
```

任何修改进入 ADP17。

---

# 22. Telemetry

Stage A 保存：

```text
raw C2-C3 disagreement by state region
raw disagreement by alpha
gamma, delta
bridge train loss
bridge val/cross-state/cross-v/cross-base R²
state-dependent bridge outputs
```

Stage B 保存：

```text
per-epoch:
C2-source train NRMSE
C3-source train NRMSE
C2 val NRMSE
C3 val NRMSE
union val NRMSE
IID/state/v/base NRMSE
a_U,b_U
```

B3 额外保存：

```text
trunk update norm
other candidate function drift
```

---

# 23. 必须生成的图

## Figure A：C2/C3 Functional Disagreement Heatmap

横轴：alpha quantile  
纵轴：state bucket  
值：\(D_{raw}(S,\alpha)\)

## Figure B：Coordinate Rescue

比较：

```text
raw disagreement
after affine bridge
after state-dependent bridge
```

## Figure C：Fresh Union Learning Curve

```text
C2 val NRMSE
C3 val NRMSE
global val NRMSE
```

## Figure D：Model Ladder

比较：

```text
ADP15 warm-start L2
B1 fresh-single-coordinate
B2 fresh-dual-coordinate
B3 fresh+trunk
```

---

# 24. 输出目录

```text
outputs/
└── e0_adp16_c2c3_diagnostic/
    ├── stage_a/
    │   ├── disagreement/
    │   ├── affine_bridge/
    │   └── state_bridge/
    ├── stage_b/
    │   ├── B1_fresh_single_coord/
    │   ├── B2_fresh_dual_coord/
    │   ├── B3_fresh_shared_trunk/
    │   └── wrong_pair_control/
    ├── state_conditional/
    └── aggregate/
        └── final_verdict.json
```

---

# 25. 命令模板

```bash
python run_adp16.py --stage smoke

python run_adp16.py --stage functional_map
python run_adp16.py --stage affine_bridge
python run_adp16.py --stage state_bridge

python run_adp16.py --stage B1
python run_adp16.py --stage B2
python run_adp16.py --stage B3

python run_adp16.py --stage wrong_pair

python analyze_adp16.py
```

---

# 26. Stop Rules

### STOP-1

wrong-pair B1 也 PASS：

```text
STOP
```

说明 fresh union test 无 mechanism discrimination。

### STOP-2

B1 PASS：

```text
停止升级模型自由度
```

不运行 B2/B3。

### STOP-3

B2 PASS：

```text
停止 shared-trunk adaptation
```

问题已经定位到 coordinate。

### STOP-4

B3 FAIL：

```text
停止继续堆 capacity
```

下一阶段转向 conditional-function / identity formation。

---

# 27. 最终科学解释

## Case A：Affine bridge PASS

\[
\boxed{\text{C2/C3 主要是 realization gauge 不一致}}
\]

优先修 shared realization coordinate。

## Case B：Affine bridge FAIL，但 B1 PASS

\[
\boxed{\text{单一机制其实可以解释两者；ADP15 失败主要来自 inherited specialization / path locking}}
\]

优先修 identity formation / candidate reset / cross-context training。

## Case C：B1 FAIL，B2 PASS

\[
\boxed{\text{共享 mechanism function 可以统一，但单一 coordinate 无法统一}}
\]

优先修 realization coordinate learning。

## Case D：只有 B3 PASS

\[
\boxed{\text{当前 shared representation 阻碍了跨-state mechanism统一}}
\]

优先修 cross-context representation learning。

## Case E：全部 FAIL

\[
\boxed{\text{C2/C3 不是简单容量或 gauge 问题；最可能是更深的 state-conditional fragmentation / identity mismatch}}
\]

下一步进入：

\[
\boxed{\text{cross-context predictive mechanism identity formation}}
\]

---

# 28. 一句话执行标准

\[
\boxed{
\text{先测试 C2/C3 能否被 state-independent realization bridge 对齐；
再用 fresh union mechanism 排除 warm-start 路径锁定；
随后依次放松 coordinate 与 shared representation。
按最小成功自由度定位：gauge、optimization、representation，还是更深的 conditional-function mismatch。}
\]
