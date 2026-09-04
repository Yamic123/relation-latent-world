# ADP6 实验执行方案：统一 realization 坐标的形成

## 0. 实验定位

ADP5 已经显示：

- response family 显著降低 sign fragmentation；
- candidate 的 family purity 已较高；
- 但 matched functional \(R^2\) 全部为负；
- 诊断显示同一 family 内每个 response point 的自由 latent \(v_r\) 没有形成统一方向、顺序和跨 family 一致尺度。

因此 ADP6 只研究：

\[
\boxed{
\text{同一个 }Z_D\text{ 内部的 realization 如何形成统一坐标。}
}
\]

本实验暂不处理：

```text
merge / split
global mechanism count
R_D
TP reconstruction
full TP participation
```

---

## 1. 核心假设

一个高层 mechanism \(Z_D^j\) 的 realization 可以变化，但同一个 mechanism 内应该存在一个跨 family 可共享的一维坐标。

允许整体 gauge：

\[
v_{\rm learned}=a_j v_{\rm GT}+b_j,\qquad a_j\neq0.
\]

因此不要求 learned latent 数值等于 GT，只要求：

1. family 内顺序一致；
2. 跨 family 坐标一致；
3. mechanism functional identity 恢复。

---

## 2. learner-visible realization coordinate

ADP6 不允许读取 GT amplitude。

对一个 response family：

\[
U=\{(S,A_r,d_r)\}_{r=1}^{R}
\]

取 baseline action \(A_0\)，定义：

\[
\delta A_r=A_r-A_0.
\]

用 family 内 action difference 的第一主方向：

\[
q_U=\operatorname{PC1}\{\delta A_r\}.
\]

定义 observable scalar：

\[
\alpha_{U,r}=q_U^\top\delta A_r,
\]

归一化：

\[
\tilde\alpha_{U,r}
=
\frac{\alpha_{U,r}}
{\max_s|\alpha_{U,s}|+\epsilon}.
\]

后续 learner 只使用 \(\tilde\alpha\)。

---

## 3. ADP6-0：action-coordinate 审计

不训练。

hidden GT 只用于 evaluator。

计算：

```text
Spearman(alpha, GT amplitude)
Kendall tau
pairwise order accuracy
sign agreement after best global flip
```

### PASS

```text
median |Spearman| > 0.98
median |Kendall| > 0.95
median pairwise order accuracy > 0.98
```

若 FAIL：

STOP。

说明 observable action geometry 不能可靠提供 realization coordinate。

---

## 4. ADP6-A0：free-v baseline

直接复用 ADP5-A1：

\[
v_{U,r}=1.5\tanh z_{U,r}.
\]

保持：

```text
same families
same candidate architecture
same assignments
same optimizer
same dataset
```

用于与 structured-v 做严格对照。

---

## 5. ADP6-A1：固定 family assignment，只测试 structured-v

这是 ADP6 最关键的实验。

### 5.1 assignment

使用 ADP5 保存的最终 family assignment：

\[
c_U.
\]

不重新做 candidate assignment。

目的只回答：

\[
\boxed{
\text{固定 family identity 后，统一 realization coordinate 是否能恢复 functional mechanism？}
}
\]

### 5.2 structured-v

取消每点自由：

\[
v_{U,r}.
\]

改为 candidate-level shared affine map：

\[
\boxed{
v_{U,r}
=
a_{c_U}\tilde\alpha_{U,r}+b_{c_U}.
}
\]

其中：

\[
a_j,b_j
\]

在 candidate \(j\) 的所有 family 间共享。

不允许 family-specific affine 作为主实验，因为那仍会让每个 family 重定义自己的 latent gauge。

### 5.3 训练目标

固定 \(c_U\)，优化：

\[
\mathcal M_j,\quad a_j,\quad b_j.
\]

loss：

\[
L_{\rm family}
=
\frac1N
\sum_{U,r}
D_S
\left(
d_{U,r},
\mathcal M_{c_U}
\left(
S_U,
a_{c_U}\tilde\alpha_{U,r}+b_{c_U}
\right)
\right).
\]

禁止加入：

```text
GT amplitude supervision
orthogonality
independence
decorrelation
arbitrary latent regularizer
```

---

## 6. A1 数据与配置

继续 ADP5：

```text
train bases = 1024
val bases = 256
test bases = 512
K_g ∈ {1,2}
Mmax = 5
```

realization points：

```text
train:   [-1.0,-0.7,-0.4,+0.4,+0.7,+1.0]
heldout: [-0.85,-0.55,-0.3,+0.3,+0.55,+0.85]
```

训练读取的是 observable \(\tilde\alpha\)，不是 GT amplitude。

配置：

```text
rounds = 40
optimizer = AdamW
mechanism lr = 3e-4
coordinate lr = 1e-3
weight_decay = 1e-5
batch = 32 families
epochs_per_round = 4
gradient_clip = 1.0
```

---

## 7. A1 核心 telemetry

每轮保存：

```text
train response MSE
heldout-realization NRMSE

candidate a_j
candidate b_j

functional R2 matrix
matched functional R2

Spearman(alpha, inferred v)
Kendall tau
pairwise order accuracy

affine-aligned realization R2
same-alpha cross-family variance

candidate state coverage
runtime
```

---

## 8. gauge-aware 指标

### 8.1 order accuracy

对 family 中任意：

\[
\alpha_r<\alpha_s
\]

检查 inferred latent 是否保持相同顺序，允许 candidate 整体反向。

### 8.2 affine-aligned realization \(R^2\)

仅 evaluator 使用 GT。

对 candidate \(j\) 拟合：

\[
v_{\rm GT}\approx a\,v_{\rm inferred}+b.
\]

报告 affine-aligned：

\[
R^2_{\rm affine}.
\]

### 8.3 same-\(\alpha\) variance

对相同 observable \(\alpha_r\)：

\[
V_j(\alpha_r)
=
\operatorname{Var}_{U:c_U=j}v_{U,r}.
\]

该量应显著低于 ADP5 free-v。

---

## 9. A1 PASS

要求：

```text
median order accuracy > 0.95
candidate affine-aligned realization R² > 0.90
heldout-realization NRMSE < 0.15

至少 2 个 purity-high candidate functional R² > 0.70
且不能继续出现所有 matched functional R² 全负
```

最关键标准：

\[
\boxed{
\text{functional }R^2\text{ 必须相对 ADP5 明显提高。}
}
\]

如果 coordinate 指标很好但 functional \(R^2\) 仍差：

STOP。

---

## 10. Negative controls

### NC-1：shuffle \(\alpha\)

每个 family 内随机打乱 \(\tilde\alpha_r\)。

预期：

```text
order metrics collapse
functional R² 下降
heldout NRMSE 上升
```

### NC-2：family-wise random reversal

每个 family 独立随机：

\[
\tilde\alpha_U\rightarrow s_U\tilde\alpha_U,\quad s_U\in\{-1,+1\}.
\]

如果跨 family 统一方向重要，这组应明显更差。

### NC-3：family-specific affine

允许：

\[
v_{U,r}=a_U\tilde\alpha_{U,r}+b_U.
\]

这组可能 reconstruction 很好，但如果 functional \(R^2\) 比 candidate-shared affine 差，则说明跨 family 共享 gauge 是必要的。

### NC-4：free-v

直接使用 ADP5：

\[
v_{U,r}=1.5\tanh z_{U,r}.
\]

---

## 11. ADP6-A2：恢复 family assignment learning

只有 A1 PASS 后执行。

candidate \(j\) 对 family \(U\) 的代价：

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
\right).
\]

assignment：

\[
c_U=\arg\min_jE(U,j).
\]

此时不再有 per-point latent inner optimization。

### 每轮

```text
1. freeze M_j,a_j,b_j
2. assign each family
3. freeze assignments
4. update M_j,a_j,b_j
5. evaluate heldout realization
6. evaluate functional identity
```

仍然：

```text
Mmax = 5
merge = off
global count cost = 0
```

---

## 12. A2 指标

```text
family Hungarian accuracy
family fragmentation
assignment change rate

functional R2 matrix
matched functional R2

order accuracy
affine-aligned realization R2
same-alpha variance

candidate usage
heldout-realization NRMSE
```

### A2 PASS

```text
family Hungarian accuracy > 0.85
mean fragmentation < 0.20
last-3 assignment change < 0.15

三个 GT 各自 best functional R² > 0.75
至少两个 > 0.90

affine-aligned realization R² > 0.90
median order accuracy > 0.95
```

---

## 13. ADP6-A3：duplicate candidate merge / global count

只有 A2 PASS 后执行。

对 candidate \(j,k\)：

训练临时 merged mechanism：

\[
\mathcal M_{j\oplus k}
\]

以及 shared coordinate：

\[
v=a_{j\oplus k}\alpha+b_{j\oplus k}.
\]

按 base 做 heldout validation。

比较：

\[
L_{\rm sep}^{heldout}
\]

和：

\[
L_{\rm merge}^{heldout}.
\]

加入 global mechanism cost：

\[
\lambda_G.
\]

接受 merge：

\[
L_{\rm merge}^{heldout}
-
L_{\rm sep}^{heldout}
<
\lambda_G.
\]

### A3 PASS

```text
alive candidate count = 3
三个 matched functional R² > 0.90
family accuracy > 0.90
false merge = 0
affine realization R² > 0.90
```

---

## 14. ADP6-B：回到完整 TP

只有 A3 PASS 后执行。

冻结学出的：

\[
Z_D=\{\mathcal M_j\}
\]

和 candidate realization coordinate。

对完整 TP：

\[
(S_i,p_i,\Delta S_i)
\]

枚举 support。

每个 active mechanism 不直接优化自由 \(v_i^j\)，而优化 scalar：

\[
\alpha_i^j
\]

然后：

\[
v_i^j=a_j\alpha_i^j+b_j.
\]

保留 discovery 阶段形成的 realization gauge。

### B PASS

```text
IID NRMSE < 0.05
101 NRMSE < 0.10
participation F1 > 0.90
min instance-effect R² > 0.90
min functional R² > 0.90
```

---

## 15. ADP6-C：TP reconstruction bridge

只有 B PASS 后执行。

比较：

\[
L_{\rm effect}
+
\lambda_PL_{\rm participation}
\]

与：

\[
L_{\rm effect}
+
\lambda_PL_{\rm participation}
+
\lambda_TL_{\rm TP-rec}.
\]

重点检查：

\[
\boxed{
L_{\rm TP-rec}
\text{ 是否保持 transformation information，同时不破坏 }Z_D\text{ identity 与 realization gauge。}
}
\]

---

## 16. 输出目录

```text
outputs/
└── e0_adp6/
    ├── adp6_0_action_coordinate/
    ├── a0_free_v_baseline/
    ├── a1_fixed_assignment_structured_v/
    ├── a2_joint_family_coordinate/
    ├── a3_merge/
    ├── b_full_tp/
    ├── c_tp_rec/
    └── negatives/
```

---

## 17. 每轮必须保存

```text
round

train_response_mse
heldout_realization_nrmse

candidate_a
candidate_b

functional_r2_matrix
matched_functional_r2

family_accuracy
fragmentation
assignment_change

order_accuracy
spearman_alpha_v
kendall_alpha_v
affine_aligned_realization_r2
same_alpha_cross_family_variance

candidate_usage
candidate_family_count

runtime
```

---

## 18. 推荐命令模板

```bash
python run_adp6.py --stage adp6_0 --seed 0

python run_adp6.py --stage adp6_a0 --seed 0
python run_adp6.py --stage adp6_a1 --seed 0

python run_adp6.py --stage adp6_nc_shuffle_alpha --seed 0
python run_adp6.py --stage adp6_nc_family_flip --seed 0
python run_adp6.py --stage adp6_nc_family_affine --seed 0
python run_adp6.py --stage adp6_nc_free_v --seed 0

python run_adp6.py --stage adp6_a2 --seed 0
python run_adp6.py --stage adp6_a3 --seed 0
python run_adp6.py --stage adp6_b --seed 0
```

---

## 19. 最重要的 A1 对照表

最终必须生成：

| 方法 | family assignment | realization 参数化 | heldout NRMSE | affine latent R² | functional R² |
|---|---|---|---:|---:|---|
| ADP5 free-v | fixed/learned | per-point free |  |  |  |
| ADP6 shared affine | fixed | candidate-shared affine |  |  |  |
| NC shuffle alpha | fixed | wrong shared affine |  |  |  |
| NC family flip | fixed | inconsistent direction |  |  |  |
| NC family affine | fixed | family-specific affine |  |  |  |

---

## 20. 失败解释

### 情况 1：ADP6-0 FAIL

action geometry 不能可靠给出 observable realization coordinate。

下一步应改为从 TP response 本身估计局部 ordering。

### 情况 2：A1 coordinate metrics 很好，但 functional R² 仍差

说明 realization coordinate 不是主要瓶颈。

不要进入 A2。

### 情况 3：A1 functional 明显恢复，但 A2 再次崩溃

说明 coordinate 已解决，但 identity assignment dynamics 仍不稳定。

下一步只研究 assignment stabilization。

### 情况 4：A2 identity 与 coordinate 都好，但仍有 4/5 candidates

说明问题 A 基本解决，只剩 global count。

进入 A3。

### 情况 5：A3 count=3 但 functional 下降

merge criterion 错误，不算成功。

---

## 21. 与整体架构的对应

ADP6 不是把 GT \(v\) 给模型。

最终完整架构中：

\[
\alpha
\]

来自：

\[
\boxed{
\text{相近状态下，沿同一局部 intervention direction 的 observable action coordinate。}
}
\]

它只提供：

```text
相对方向
相对顺序
相对强度
```

而 \(Z_D\) 学习：

\[
\boxed{
\text{这种 intervention variation 对世界产生的高层 response rule。}
}
\]

因此：

\[
A\neq Z_D.
\]

但 action geometry 可以帮助定义：

\[
v_i^j
\]

在同一个 \(Z_D^j\) 内部应该如何有序变化。

---

## 22. 一句话执行标准

\[
\boxed{
\text{先固定 ADP5 已形成的 family identity，}
}
\]

\[
\boxed{
\text{只改变 }v\text{ 的参数化，验证统一 realization coordinate 是否让 functional }Z_D\text{ 真正出现。}
}
\]

只有这一步成立，才允许恢复 assignment learning、merge 和完整 TP。
