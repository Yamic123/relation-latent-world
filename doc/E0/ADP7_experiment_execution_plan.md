# ADP7 实验执行方案：跨低层变化稳定性的功能合并

## 0. 实验定位

ADP6 已经证明：统一 realization coordinate 后，candidate function 可以恢复到近乎完美；但恢复 assignment learning 后，同一真实机制仍存在 duplicate candidates，例如：

```text
GT1 -> C2 / C5
GT3 -> C1 / C4
GT2 -> C3
```

因此 ADP7 只研究：

\[
\boxed{
\text{两个 duplicate candidate 是否真的可以被一个更高层、低层变化下稳定的 }Z_D\text{ 所替代。}
}
\]

重点不是“两个 candidate 能否在 IID validation 上合并”，而是：

\[
\boxed{
\text{合并后是否跨 state、realization、base/TP 变化仍保持同一高层规律。}
}
\]

---

## 1. 核心假设

一个合法 merge：

\[
C_j,C_k \rightarrow C_{jk}
\]

必须同时满足：

\[
\boxed{
\text{功能等价}
+
\text{跨低层变化稳定}
+
\text{不依赖额外容量作弊}
}
\]

即：

1. merged mechanism 能解释两边数据；
2. 在未见低层条件下仍成立；
3. merged model 容量与单个 candidate 相同；
4. realization coordinate 仍然是一套共享坐标。

---

## 2. 禁止的退化 merge

禁止：

```text
拼接两个 candidate 网络
增加 hidden dim / heads / layers
为两边保留两套内部 coordinate
用 GT label 决定 merge
只看 IID validation
只看参数距离
只看 usage overlap
```

merged mechanism 必须是一个单-candidate 容量的：

\[
\mathcal M_{jk}
\]

并且只使用一套：

\[
v=a_{jk}\alpha+b_{jk}.
\]

---

## 3. 起点

使用 ADP6-A2 最终 checkpoint。

固定：

```text
world
dataset
response family construction
observable alpha
candidate architecture
candidate-level coordinate form
```

ADP7 不再修改：

```text
representation
family construction
realization coordinate definition
R_D
TP-rec
full TP participation
```

---

## 4. 实验对象

development seed 0 首先测试 suspected duplicate pairs：

```text
P1 = C2 + C5
P2 = C1 + C4
```

并测试错误 pair negative controls，例如：

```text
C2 + C3
C3 + C4
C1 + C2
```

GT identity 只允许 evaluator 使用。

---

## 5. separate 与 merged 模型

对 pair \(C_j,C_k\)：

### Separate

\[
L_{sep}=L_j+L_k
\]

各自保留：

\[
v_j=a_j\alpha+b_j,\qquad
v_k=a_k\alpha+b_k.
\]

### Merged

训练全新：

\[
\mathcal M_{jk}
\]

以及共享坐标：

\[
v=a_{jk}\alpha+b_{jk}.
\]

训练数据：

\[
\mathcal D_{jk}=\mathcal D_j\cup\mathcal D_k.
\]

目标：

\[
L_{merge}
=
\frac1N
\sum_{U\in\mathcal D_{jk}}
\sum_r
D_S
\left(
d_{U,r},
\mathcal M_{jk}
\left(
S_U,
a_{jk}\alpha_{U,r}+b_{jk}
\right)
\right).
\]

---

## 6. Stage

```text
ADP7-0  duplicate pair 审计
   ↓
A0      IID heldout merge sanity
   ↓
A1      跨 state holdout
   ↓
A2      跨 realization holdout
   ↓
A3      跨 base/TP holdout
   ↓
A4      联合低层变化 holdout
   ↓
A5      正式 merge + global count
```

任一步 FAIL：STOP。

---

## 7. ADP7-0：duplicate pair 审计

不训练。

对每对 candidate 计算：

```text
cross-state functional probe similarity
heldout family prediction similarity
|a_j-a_k|
|b_j-b_k|
S-space coverage overlap
```

evaluator-only：

```text
GT identity agreement
GT functional R²
```

目的不是直接 merge，而是确认 pair 是 learner 视角下的高疑似 duplicate。

---

## 8. A0：IID heldout merge sanity

按 base 随机拆：

```text
train = 70%
heldout = 30%
```

训练 merged model。

比较：

\[
L_{sep}^{heldout}
\]

与：

\[
L_{merge}^{heldout}.
\]

定义：

\[
\Delta_{IID}
=
\frac{
L_{merge}^{heldout}
-
L_{sep}^{heldout}
}{
L_{sep}^{heldout}+\epsilon
}.
\]

### PASS

```text
Δ_IID < 0.05
merged functional R² > 0.95
affine realization R² > 0.95
```

A0 只能作为 sanity check，不能作为正式 merge 依据。

---

## 9. A1：跨 state 稳定性

目的：检查 merged \(Z_D\) 是否只是记住已见状态区域。

对 base state \(S\) 做 PCA：

\[
z_S=PC1(S).
\]

第一组：

```text
train = middle 60%
test-low = lowest 20%
test-high = highest 20%
```

第二组反向：

```text
train = low + high
test = middle
```

分别比较 separate / merged：

```text
NRMSE
functional R²
coordinate affine R²
```

定义：

\[
\Delta_S
=
NRMSE_{merge}
-
NRMSE_{sep}.
\]

### PASS

```text
Δ_S < 0.03
merged functional R² > 0.90
coordinate affine R² > 0.95
两种 state split 都通过
```

---

## 10. A2：跨 realization 稳定性

至少做三种 split。

### A2-1 弱 -> 强

```text
train: |alpha| <= 0.7
test:  |alpha| > 0.7
```

### A2-2 强 -> 弱

```text
train: |alpha| >= 0.7
test:  |alpha| < 0.7
```

### A2-3 单侧 -> 另一侧

```text
train alpha < 0 -> test alpha > 0
train alpha > 0 -> test alpha < 0
```

### PASS

每种 split：

```text
merged NRMSE <= separate NRMSE + 0.03
functional R² > 0.90
order accuracy > 0.95
affine realization R² > 0.95
```

全部通过才 PASS。

---

## 11. A3：跨 base / TP 稳定性

按 base 完全不重叠：

```text
train A = 50%
test B = 50%
```

再交换：

```text
train B -> test A
```

任何 family 不允许跨 split 泄露。

### PASS

```text
双向 Δ_base < 0.03
functional R² > 0.90
heldout NRMSE < 0.05
```

---

## 12. A4：联合低层变化 holdout

同时让 test 条件在：

```text
state
realization
base
```

三个维度都偏离训练集。

示例：

```text
train:
middle-S
negative / weak alpha
base subset A

test:
extreme-S
positive / strong alpha
base subset B
```

再做互换方向。

### PASS

```text
merged functional R² > 0.85
merged NRMSE <= separate NRMSE + 0.05
coordinate affine R² > 0.90
order accuracy > 0.95
```

而错误 pair 必须明显失败。

---

## 13. Negative controls

### NC-1：容量翻倍假 merge

构造：

```text
hidden dim × 2
或
heads × 2
```

证明“更大模型可以吞掉两边”不等价于高层 identity。

主实验不允许用这组模型。

### NC-2：只做 IID validation

若：

```text
IID PASS
structured holdout FAIL
```

则证明普通 validation 会产生伪 merge。

### NC-3：family-specific coordinate

允许：

\[
v=a_U\alpha+b_U.
\]

若 reconstruction 好但 structured holdout 差，说明 family-specific gauge 在隐藏退化解。

### NC-4：错误 pair merge

至少测试：

```text
C2+C3
C3+C4
```

要求 structured holdout 明显失败。

---

## 14. A5：正式 merge

只有 A0-A4 全部 PASS 后执行。

定义结构化损失：

\[
L_{structured}
=
w_S L_S
+
w_V L_V
+
w_B L_B
+
w_J L_J
\]

第一版：

```text
w_S = 0.25
w_V = 0.25
w_B = 0.20
w_J = 0.30
```

比较：

\[
J_{sep}
=
L_{structured}^{sep}
+
2\lambda_G
\]

\[
J_{merge}
=
L_{structured}^{merge}
+
\lambda_G.
\]

只有：

\[
J_{merge}<J_{sep}
\]

才正式 merge。

---

## 15. lambda_G sweep

development seed 0：

```text
lambda_G ∈ {1e-4,3e-4,1e-3,3e-3,1e-2}
```

不能根据“最终是否恰好 3 个”来选。

选择规则：

```text
在 structured loss 距最优 <= 2% 的配置中，
选择 alive candidate 数最少者。
```

---

## 16. 正式 merge 后

若：

```text
C2 + C5 -> C25
C1 + C4 -> C14
```

则 alive：

```text
C25
C14
C3
```

总数为 3。

随后重新做一次 family assignment，但禁止再次拆分 merged candidate。

### A5 PASS

```text
alive mechanism count = 3
family Hungarian accuracy > 0.90
mean fragmentation < 0.10
assignment change < 0.10

三个 functional R² > 0.90

Cross-S PASS
Cross-v PASS
Cross-base PASS
Joint PASS
```

---

## 17. 初始化

merged model 两种初始化都跑：

```text
Init-A: random
Init-B: better source candidate copy
```

两种都能通过更可信。

---

## 18. 推荐训练配置

沿用 ADP6：

```text
optimizer = AdamW
mechanism lr = 3e-4
coordinate lr = 1e-3
weight_decay = 1e-5
batch = 32 families
epochs = 40
gradient_clip = 1.0
```

---

## 19. 每个 pair 必须记录

```text
pair id

train loss
separate heldout NRMSE
merged heldout NRMSE
delta NRMSE

functional R²

coordinate a,b
affine realization R²
order accuracy

state-region metrics
realization-region metrics
base metrics
joint metrics

parameter count
runtime
```

---

## 20. 输出目录

```text
outputs/
└── e0_adp7/
    ├── adp7_0_pair_audit/
    ├── a0_iid_merge/
    ├── a1_state_holdout/
    ├── a2_realization_holdout/
    ├── a3_base_holdout/
    ├── a4_joint_holdout/
    ├── a5_formal_merge/
    └── negatives/
        ├── double_capacity/
        ├── iid_only/
        ├── family_specific_coordinate/
        └── wrong_pairs/
```

---

## 21. 推荐命令模板

```bash
python run_adp7.py --stage adp7_0 --seed 0

python run_adp7.py --stage adp7_a0 --pair C2,C5 --seed 0
python run_adp7.py --stage adp7_a0 --pair C1,C4 --seed 0

python run_adp7.py --stage adp7_a1 --pair C2,C5 --seed 0
python run_adp7.py --stage adp7_a1 --pair C1,C4 --seed 0

python run_adp7.py --stage adp7_a2 --pair C2,C5 --seed 0
python run_adp7.py --stage adp7_a2 --pair C1,C4 --seed 0

python run_adp7.py --stage adp7_a3 --pair C2,C5 --seed 0
python run_adp7.py --stage adp7_a3 --pair C1,C4 --seed 0

python run_adp7.py --stage adp7_a4 --pair C2,C5 --seed 0
python run_adp7.py --stage adp7_a4 --pair C1,C4 --seed 0

python run_adp7.py --stage adp7_a5 --seed 0
```

---

## 22. 最重要结果表

| Pair | IID | Cross-S | Cross-v | Cross-base | Joint | Functional R² | Merge? |
|---|---:|---:|---:|---:|---:|---:|---|
| C2+C5 |  |  |  |  |  |  |  |
| C1+C4 |  |  |  |  |  |  |  |
| wrong pair 1 |  |  |  |  |  |  |  |
| wrong pair 2 |  |  |  |  |  |  |  |

---

## 23. 失败解释

### IID PASS，但 structured holdout FAIL

说明这是已知数据上的退化 merge，不是稳定 \(Z_D\)。

### Cross-S FAIL

说明两个 candidate 可能分别覆盖不同 state regime。

### Cross-v FAIL

说明两个 candidate 可能分别承担不同 realization regime。

### Cross-base FAIL

说明 merge 依赖具体 interaction instance。

### 正确 pair PASS，错误 pair 也 PASS

说明测试没有区分力，不能进入正式 merge。

### 只有 double-capacity merge PASS

说明合并依赖额外容量，是明显退化解。

---

## 24. 与高层 Z_D 定义的对应

ADP7 的 merge 判据不是：

\[
\boxed{
\text{两组数据能不能塞进同一个网络}
}
\]

而是：

\[
\boxed{
\text{当低层 state、realization、interaction instance 改变后，
同一个高层 transformation rule 是否仍然成立。}
}
\]

因此合法 merge 实际上是在验证：

\[
\boxed{
Identity(Z_D)
\text{ 是否在低层变化下保持稳定。}
}
\]

---

## 25. 一句话执行标准

\[
\boxed{
\text{只有一个单-candidate 容量的 merged mechanism，
在跨 state、跨 realization、跨 base、联合低层变化下
都能无损替代两个 duplicate candidate，才允许 merge。}
}
\]
