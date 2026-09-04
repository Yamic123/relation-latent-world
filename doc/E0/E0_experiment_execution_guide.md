# E0 实验执行指导：Relational Latent World Learning 的可实现性、可辨识性与优化恢复测试

> **文档定位**  
> 本文档面向完全不了解项目背景的工程师。工程师应能够仅依据本文档完成 E0 实验的数据生成、模型训练、评估、结果可视化与结论判定。  
> E0 不是机器人实验，也不验证真实世界是否真的具有本文提出的潜变量结构。E0 只回答一个更基础的问题：
>
> \[
> \boxed{
> \text{如果数据世界严格满足我们提出的 }W_D=(Z_D,R_D)\text{ 数学结构，且数据具有足够辨识信息，}
> }
> \]
> \[
> \boxed{
> \text{当前学习目标、网络参数化与优化程序能否从观测到的数据中把该结构恢复出来？}
> }
> \]

---

# 1. E0 的科学目的

完整理论希望从 interaction 中发现：

\[
Interaction
\rightarrow TP
\rightarrow Z_D
\leftrightarrow R_D
\rightarrow W_D.
\]

其中：

- \(TP\)：Transformation Primitive，局部 transformation 的 latent representation；
- \(Z_D=\{Z_D^j\}\)：domain-level reusable effective variables；
- \(R_D=\{R_D^{j\rightarrow k}\}\)：effective variables 之间稳定、方向性的 modulation relation；
- \(W_D=(Z_D,R_D)\)：relational latent world。

E0 暂时**不验证**：视觉编码、Slot Attention、TP discovery、TP boundary segmentation、exploration、\(C_T\)、task execution、真实机器人控制。

E0 将这些问题全部替换为人工可控条件，只保留：

\[
TP\rightarrow Z_D\leftrightarrow R_D.
\]

因此 E0 本质是：

\[
\boxed{
Realizability + Identifiability\ Coverage + Optimization
}
\]

测试。

---

# 2. 为什么 E0 必须先做

如果直接进入完整系统，最终 \(W_D\) 学习失败，可能来自：

1. 视觉 representation 错误；
2. TP 错误；
3. interaction data 缺少足够 variation；
4. \(Z_D/R_D\) 数学目标不可辨识；
5. 网络容量不足；
6. optimizer 没找到正确解；
7. implementation bug。

E0 主动移除前 1--3 类因素。

如果 E0 的人工世界严格由我们的结构方程生成，并且 oracle solution 能达到零误差，但 learner 仍恢复失败，则问题只能落在：

\[
\boxed{
objective,\ parameterization,\ optimization,\ implementation,\ identifiability.
}
\]

---

# 3. E0 的实验分层

## E0-A：只验证 \(Z_D\)

真实世界没有 relation：

\[
R_D=\varnothing.
\]

数据严格由 reusable mechanisms 相加生成：

\[
\boxed{
\Delta S=\sum_j e^j.
}
\]

目标：learner 是否能从一个没有 mechanism label 的 TP representation 中恢复 reusable mechanism decomposition？

## E0-B：加入一个单向 relation

在已经通过 E0-A 的 world generator 上加入唯一真实边：

\[
\boxed{R_D^{1\rightarrow2}.}
\]

数据由：

\[
\boxed{
\Delta S
=
\sum_j e^j
+
g_{12}r^{1\rightarrow2},
\qquad
g_{12}=1.
}
\]

生成。

目标：在 individual mechanisms 已经可恢复的条件下，learner 是否能从 residual 中恢复正确 relation，并区分 \(1\rightarrow2\) 与 \(2\rightarrow1\)？

---

# 4. Ground-truth synthetic world

第一版固定：

\[
K=3,
\qquad
d_s=2,
\qquad
M^*=3,
\qquad
M_{\max}=5,
\qquad
d_v=1,
\qquad
d_p=6.
\]

因此：

\[
S=\{s^1,s^2,s^3\}\in\mathbb R^{3\times2}.
\]

learner 不被告知“世界中恰好有三个 mechanisms”，只知道最多有五个 candidate mechanisms。

---

# 5. State 生成

每个 sample 独立采样：

\[
s^k\sim U([-1,1]^2),
\qquad k=1,2,3.
\]

代码逻辑：

```python
S = Uniform(-1, 1).sample((3, 2))
```

第一版不加入 observation noise。E0 首先测试结构恢复本身，不应把 noise robustness 与优化正确性混在一起。

---

# 6. Mechanism participation 生成

真实 participation：

\[
m=(m_1,m_2,m_3)\in\{0,1\}^3.
\]

第一版采用 **balanced factorial sampling**。

理论上共有：

\[
2^3=8
\]

种 pattern：

\[
000,001,010,011,100,101,110,111.
\]

建议 E0-A 训练使用：

\[
001,010,011,100,110,111
\]

以及少量 \(000\)。

**训练中完全不出现：**

\[
\boxed{101}
\]

作为 compositional test。

这样 \(Z_1\) 和 \(Z_3\) 分别在其它 context 中出现，但从未共同出现。

---

# 7. Mechanism realization 生成

若 \(m_j=1\)，采样：

\[
v_j
\sim
U([-1,-0.3]\cup[0.3,1]).
\]

若 \(m_j=0\)，令：

\[
v_j=0.
\]

第一版有意避开 active mechanism 的 \(v_j\approx0\)，因为：

\[
m_j=1,\ v_j\approx0
\]

可能与：

\[
m_j=0
\]

在 effect 上变得难以区分。E0 第一阶段不主动制造这一额外 ambiguity。

---

# 8. Mechanism 对 slot 的 ground-truth support

固定：

\[
\boxed{
C=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&0&1
\end{bmatrix}.
}
\]

row 为 mechanism \(j\)，column 为 slot \(k\)。

含义：

- \(Z_1\) 作用于 slot 1 和 slot 2；
- \(Z_2\) 作用于 slot 2 和 slot 3；
- \(Z_3\) 作用于 slot 1 和 slot 3。

该设计故意让不同 mechanisms 的 slot support 重叠，避免 learner 通过：

\[
Z_j\leftrightarrow slot_j
\]

这种 trivial shortcut 恢复答案。

---

# 9. Ground-truth mechanism function

定义 mechanism \(j\) 对 slot \(k\) 的 effect：

\[
\boxed{
e_j^k
=
m_j C_{jk}
\left[
v_jA_{jk}s^k
+
v_jb_{jk}
+
\alpha v_j^2c_{jk}
\right]
}
\]

其中第一版：

\[
\alpha=0.25.
\]

参数维度：

\[
A_{jk}\in\mathbb R^{2\times2},
\]

\[
b_{jk}\in\mathbb R^2,
\qquad
c_{jk}\in\mathbb R^2.
\]

整个 mechanism effect：

\[
e_j=\{e_j^1,e_j^2,e_j^3\}.
\]

| 项 | 作用 |
|---|---|
| \(m_j\) | 当前 interaction 中 mechanism \(j\) 是否参与 |
| \(C_{jk}\) | mechanism \(j\) 是否直接作用于 slot \(k\) |
| \(v_jA_{jk}s^k\) | state-dependent realization effect |
| \(v_jb_{jk}\) | state-independent realization component |
| \(0.25v_j^2c_{jk}\) | mild nonlinear dependence on realization |

该公式不是现实物理定律，只是一个：

\[
\boxed{
simple\ enough\ to\ diagnose,
\quad
nontrivial\ enough\ to\ prevent\ shortcuts
}
\]

的 synthetic ground-truth mechanism family。

---

# 10. Ground-truth 参数初始化

固定一个 **world seed**：

```text
WORLD_SEED = 20260901
```

对所有 \(C_{jk}=1\) 的 pair：

```python
A_jk ~ Normal(0, 1)
b_jk ~ Normal(0, 0.5)
c_jk ~ Normal(0, 0.5)
```

然后对 \(A_{jk}\) 做尺度归一化，例如：

```python
A_jk = A_jk / spectral_norm(A_jk)
```

初始化后永久保存：

```text
ground_truth_world.pt
```

同一个 E0 系列中的不同 optimizer seeds 不得重新生成 world。

必须区分：

- world seed；
- dataset seed；
- optimization seed。

---

# 11. E0-A 的 world equation

E0-A 不包含 relation。

对每个 sample：

\[
\boxed{
\Delta S
=
e_1+e_2+e_3.
}
\]

并可记录：

\[
S^{post}=S^{pre}+\Delta S.
\]

---

# 12. Oracle TP 构造

E0 不测试 TP discovery，因此人为提供一个**完美 TP**。但 TP 不能直接暴露 \(m_j,v_j\)。

至少依次执行两种 TP representation。

## TP-0：Identity TP

\[
\boxed{
p=\operatorname{vec}(\Delta S).
}
\]

目的：最简单 realizability sanity check。

## TP-1：Random orthogonal TP

生成固定随机 orthogonal matrix：

\[
B\in\mathbb R^{6\times6},
\qquad
B^\top B=I.
\]

然后：

\[
\boxed{
p=B\operatorname{vec}(\Delta S).
}
\]

选择 orthogonal matrix 的原因：

1. \(B^{-1}=B^\top\)，无信息损失；
2. \(\|Bx\|_2=\|x\|_2\)，不改变尺度；
3. condition number \(\kappa(B)=1\)，不引入数值病态；
4. dense rotation 会混合原始 slot/change coordinates，减少 coordinate shortcut。

因此如果 identity TP 成功而 orthogonal TP 失败，应认为 learner 对 TP coordinate system 存在严重依赖，E0 不应完全通过。

---

# 13. Learner 可见与不可见的信息

## Learner 可见

每个 sample：

\[
\boxed{
(S_i,p_i,\Delta S_i).
}
\]

## Learner 不可见

训练期间禁止使用：

\[
m_i^*,
\quad
v_i^*,
\quad
C,
\quad
A_{jk}^*,
\quad
b_{jk}^*,
\quad
c_{jk}^*,
\quad
R_D^*.
\]

推荐 dataset 保存两份：

```text
visible_data.npz
ground_truth_labels.npz
```

训练 dataloader 只允许读取第一份。

---

# 14. Exact Realizability Certificate

在训练 learner 以前，必须执行 oracle check。

使用真实：

\[
m^*,v^*,\mathcal M_j^*
\]

重新计算：

\[
\widehat{\Delta S}^{oracle}
=
\sum_jm_j^*\mathcal M_j^*(S,v_j^*).
\]

计算：

\[
\boxed{
L_{oracle}
=
\frac1N
\sum_i
\left\|
\Delta S_i-
\widehat{\Delta S}_i^{oracle}
\right\|_2^2.
}
\]

无噪声 E0-A 要求：

\[
\boxed{L_{oracle}<10^{-10}.}
\]

如果失败：

\[
\boxed{STOP.}
\]

不得继续 learner training。此时 generator 与理论公式/实现不一致，实验无效。

---

# 15. E0-A Learner

学习：

\[
\boxed{q_\eta(m,v\mid p,S)}
\]

和 candidate mechanism bank：

\[
\{\mathcal M_j\}_{j=1}^{M_{\max}}.
\]

第一版：

\[
M_{\max}=5.
\]

---

# 16. Mechanism-query abstraction network

为每个 candidate mechanism 建立 learned query：

\[
q_D^j.
\]

输入 context：

\[
X_i=
\{E_p(p_i),E_s(s_i^1),E_s(s_i^2),E_s(s_i^3)\}.
\]

每个 query：

\[
h_i^j=CrossAttn(q_D^j,X_i).
\]

从 \(h_i^j\) 预测：

\[
m_i^j,\quad v_i^j.
\]

第一版建议：

```text
token dim: 64
attention heads: 4
cross-attention layers: 2
M_max: 5
```

---

# 17. Participation head

使用 Hard-Concrete：

\[
m_i^j\sim HardConcrete(\ell_i^j).
\]

训练使用 relaxed sample，评估时使用 expected gate probability 或 deterministic gate。

complexity：

\[
\boxed{
L_{participation}
=
\sum_{i,j}
\mathbb E[\mathbf 1(m_i^j\neq0)].
}
\]

---

# 18. Realization head

E0 第一版建议优先使用 deterministic scalar realization：

\[
\boxed{v_i^j=H_v(h_i^j)}
\]

而不是立刻使用 stochastic Gaussian。

原因：E0 首轮需要减少随机优化因素，先验证 decomposition 本身。

当 deterministic E0 稳定后，再替换为正式模型中的 Gaussian head：

\[
q(v_j)=\mathcal N(\mu_j,\sigma_j^2).
\]

---

# 19. Learned mechanism bank

正式语义：

\[
e_i^j
=
m_i^j\mathcal M_j(S_i,v_i^j).
\]

E0 第一版保持：

\[
\boxed{
shared\ set\ backbone
+
mechanism\text{-}specific\ adapter.
}
\]

推荐最小实现：

1. 每个 slot 用 MLP 编码；
2. mechanism query + \(v_j\) 产生 condition；
3. 使用 FiLM / adaptive affine modulation；
4. set-equivariant self-attention；
5. 每个 slot 输出 2D effect。

learner 不知道 ground-truth incidence matrix \(C\)。

---

# 20. E0-A loss

第一轮只使用：

\[
\boxed{
L_{Z_D}^{E0-A}
=
L_{effect}^D
+
\lambda_P L_{participation}.
}
\]

其中：

\[
L_{effect}^D
=
\frac1N
\sum_i
D_S
\left(
\Delta S_i,
\sum_j e_i^j
\right).
\]

建议第一轮：

\[
D_S=MSE.
\]

不加入 \(L_{TP-rec}\)。

---

# 21. E0-A 数据规模

第一版不测试 sample efficiency。

推荐：

```text
train: 100,000
validation: 20,000
test_iid: 20,000
test_context: 20,000
test_combination_101: 20,000
```

---

# 22. E0-A Train/Test 设计

## Test-IID

与训练相同 support，但独立 sample。

## Test-Context

改变 \(S,v\) joint combinations。第一版优先做 interpolation-based held-out context，避免把 extrapolation difficulty 与 mechanism recovery 混为一谈。

## Test-Combination

训练完全不出现：

\[
\boxed{m=101.}
\]

测试全部为：

\[
m=101.
\]

这是最关键 compositional test。

---

# 23. E0-A 评估原则

低：

\[
L_{effect}
\]

只能证明 learner 可以预测 \(\Delta S\)，不能证明找到了真实 mechanism decomposition。

必须同时评价：

1. predictive recovery；
2. participation recovery；
3. mechanism functional recovery；
4. compositional recovery；
5. optimization stability。

---

# 24. Prediction metric

推荐 world NRMSE：

\[
\boxed{
NRMSE_W
=
\frac{
\sqrt{\mathbb E\|\Delta S-\widehat{\Delta S}\|^2}
}{
\sqrt{\mathbb E\|\Delta S-\mathbb E[\Delta S]\|^2}
}.
}
\]

分别报告：train、validation、test-IID、test-context、held-out combination。

---

# 25. Permutation alignment

candidate mechanism index 没有语义顺序。

learned：

\[
\hat Z_1
\]

可能对应 GT：

\[
Z_3.
\]

因此所有 structure metrics 必须先进行 permutation alignment。

先建立 similarity matrix：

\[
Sim(l,j).
\]

再用 Hungarian algorithm：

\[
\pi^*
=
\argmax_\pi
\sum_l
Sim(l,\pi(l)).
\]

只匹配 3 个 active learned mechanisms 与 3 个 GT mechanisms，多余 learner mechanisms 视为 redundant candidates。

---

# 26. Mechanism functional recovery

不要比较网络参数。

在大量独立 probe：

\[
(S^{probe},v_j^{probe})
\]

上计算 GT effect：

\[
e_j^*=\mathcal M_j^*(S,v_j).
\]

learned \(v_l\) 可能与 GT \(v_j\) 存在 affine gauge，因此先在 calibration set 上拟合：

\[
v_l\approx a_jv_j+b_j.
\]

然后测试 learned mechanism effect 与 GT effect 的一致性。

推荐：

\[
\boxed{
R^2_{func}(l,j)
=
1-
\frac{
\mathbb E\|\hat e_l-e_j^*\|^2
}{
\mathbb E\|e_j^*-\bar e_j^*\|^2
}.
}
\]

functional similarity 用于 Hungarian matching。

---

# 27. Participation recovery

完成 permutation alignment 后，比较：

\[
\hat m_i^j
\]

与：

\[
m_i^{j*}.
\]

报告：

- Precision；
- Recall；
- F1；
- AUROC。

如果 world prediction 很好但 participation recovery 很差，说明模型并未恢复我们人工构造的 mechanism decomposition。

---

# 28. Redundant candidate evaluation

因为：

\[
M_{\max}=5>M^*=3,
\]

需要报告多余 candidate usage：

\[
u_j
=
\frac1N\sum_i\mathbb E[m_i^j].
\]

理想：

\[
u_{redundant}\rightarrow0.
\]

如果五个 mechanisms 都高度使用但 prediction 很好，说明 decomposition 存在 redundancy，E0 不应判为完全通过。

---

# 29. E0-B：Ground-truth relation

E0-A 通过后，保持同一 mechanism generator。

加入唯一真实 relation：

\[
\boxed{R_D^{1\rightarrow2}.}
\]

GT edge：

\[
g_{12}=1,
\]

其余：

\[
g_{jk}=0.
\]

---

# 30. Relation generator

先计算 target baseline \(e_2\)。

定义：

\[
\gamma^{12}
=
a_\gamma\tanh(v_1)h_\gamma(S),
\]

\[
\beta^{12}
=
a_\beta\tanh(v_1v_2)b_R.
\]

第一版将：

\[
h_\gamma(S)
=
\tanh(w_R^\top\operatorname{mean}(S)).
\]

推荐：

\[
a_\gamma=0.4,
\qquad
a_\beta=0.1.
\]

relation-modified target：

\[
\tilde e_2
=
(1+\gamma^{12})\odot e_2+\beta^{12}.
\]

relation residual：

\[
\boxed{
r^{1\rightarrow2}
=
m_1m_2
(\tilde e_2-e_2).
}
\]

最终：

\[
\boxed{
\Delta S
=
e_1+e_2+e_3
+
g_{12}r^{1\rightarrow2}.
}
\]

**注意：\(g_{12}\) 只出现一次。**

---

# 31. 为什么 E0-B 需要 factorial coverage

数据中必须同时有：

\[
m_1=1,m_2=0,
\]

\[
m_1=0,m_2=1,
\]

以及：

\[
m_1=1,m_2=1.
\]

这样 learner 才能分别观察：

- \(Z_1\) main effect；
- \(Z_2\) main effect；
- 二者共同存在时额外出现的 non-additive residual。

若 \(Z_1,Z_2\) 永远一起出现，relation 和 main effect 会产生严重 gauge ambiguity。

---

# 32. E0-B Learner

Stage 1：从 E0-A checkpoint 初始化或重新训练 \(Z_D\)。

Stage 2：冻结或显著降低：

\[
q_\eta,\quad\mathcal M_j
\]

learning rate。

增加所有 directed candidate edges：

\[
(j,k),\qquad j\neq k.
\]

每个 edge：

\[
g_{jk}\sim HardConcrete(\alpha_{jk}).
\]

---

# 33. Learned relation network

采用：

\[
\boxed{
shared\ directed\ residual\ FiLM\ network
+
pair\text{-}specific\ embedding.
}
\]

对 candidate \(j\rightarrow k\)：

\[
(\gamma_i^{jk},\beta_i^{jk})
=
F_R(v_i^j,v_i^k,S_i,q_R^{j\rightarrow k}).
\]

然后：

\[
\tilde e_i^{j\rightarrow k}
=
(1+\gamma_i^{jk})\odot e_i^k
+
\beta_i^{jk},
\]

\[
\boxed{
r_i^{j\rightarrow k}
=
m_i^jm_i^k
(\tilde e_i^{j\rightarrow k}-e_i^k).
}
\]

world prediction：

\[
\boxed{
\widehat{\Delta S}_i^W
=
\sum_j e_i^j
+
\sum_{j\neq k}g_{jk}r_i^{j\rightarrow k}.
}
\]

---

# 34. E0-B relation loss

\[
\boxed{
L_R
=
L_{world}
+
\lambda_RL_{edge}^{support}.
}
\]

其中：

\[
L_{world}
=
D_S(\Delta S,\widehat{\Delta S}^{W}).
\]

E0 synthetic data 可直接保证真实 candidate pair 拥有充分 factorial support。

第一版可先令：

\[
s_{jk}^R=1
\]

用于充分证据场景。随后再做“减少 directional support”的压力测试。

---

# 35. Relation explanatory power

对每条 learned edge：

\[
j\rightarrow k
\]

在 held-out test set 上计算：

\[
\boxed{
EP_{j\rightarrow k}
=
\mathbb E
\left[
D_S(\Delta S,\widehat{\Delta S}^{-jk})
-
D_S(\Delta S,\widehat{\Delta S}^{W})
\right].
}
\]

预期：

\[
EP_{1\rightarrow2}>0
\]

且显著高于：

\[
EP_{2\rightarrow1}
\]

与其它 false edges。

---

# 36. Relation functional recovery

在 held-out：

\[
S,v_1,v_2
\]

上比较：

\[
r^{1\rightarrow2}_{GT}
\]

与 aligned learned：

\[
\hat r^{1\rightarrow2}.
\]

报告 functional \(R^2\) 或 normalized MSE。

---

# 37. Ground-truth edge recovery

synthetic E0 中可额外报告：

- directed edge precision；
- recall；
- F1；
- AUROC；
- false edge count。

但 edge gate 大小不是唯一判据，必须同时满足：

\[
\boxed{
edge\ gate
+
held\text{-}out\ EP
+
functional\ residual\ recovery.
}
\]

---

# 38. Optimization seeds

固定同一个 ground-truth world 和 dataset。

至少运行：

\[
\boxed{10\text{ independent optimization seeds}.}
\]

推荐：

```text
0,1,2,3,4,5,6,7,8,9
```

报告：

\[
\boxed{
RecoveryRate
=
\frac{\#\text{通过所有核心结构条件的 seed}}{10}.
}
\]

---

# 39. 推荐初始超参数

以下为起始值，不是理论常数。

| 参数 | 建议起始值 |
|---|---:|
| batch size | 512 |
| optimizer | AdamW |
| learning rate E0-A | \(3\times10^{-4}\) |
| relation stage LR | \(1\times10^{-4}\) |
| weight decay | \(10^{-5}\) |
| max epochs | 200 |
| gradient clip | 1.0 |
| \(\lambda_P\) | \(10^{-3},10^{-2},10^{-1}\) 小 sweep |
| \(\lambda_R\) | \(10^{-3},10^{-2},10^{-1}\) 小 sweep |
| optimization seeds | \(\ge10\) |

不要在 E0 初期进行大规模超参搜索。先确认 implementation 正确、oracle 可实现、至少存在一个可恢复区域，再测试稳定性。

---

# 40. Negative controls

E0 必须包含 negative controls。

| Control | 修改 | 预期结果 |
|---|---|---|
| NC1 | \(M_{\max}<M^*\) | structure recovery / held-out composition 明显下降 |
| NC2 | 删除 factorial mechanism variation | mechanism separation 变差 |
| NC3 | relation world 使用 no-\(R_D\) learner | joint-condition residual 持续存在 |
| NC4 | shuffle \(p\) across samples | instance mechanism allocation 显著失败 |
| NC5 | \(\lambda_P=0\) | redundant mechanism usage 上升 |
| NC6 | 删除方向辨识所需 variation | edge orientation 稳定性下降 / unresolved 增加 |
| NC7 | 使用 ill-conditioned 非正交 TP transform | optimization 更困难；用于后续压力测试 |

若 negative controls 与正式模型表现无差别，需要怀疑：指标无效、implementation bypass，或数据没有形成理论要求的压力。

---

# 41. 必须输出的图表

## Figure A：Participation heatmap

GT 与 learned \(m\)，learned 结果先做 permutation alignment。

## Figure B：Mechanism functional recovery

每个 \(Z_j\)：

\[
e_j^{GT}
\quad vs\quad
e_j^{learned}.
\]

报告 \(R^2\)。

## Figure C：Held-out combination prediction

专门展示：

\[
m=101.
\]

## Figure D：Directed relation EP matrix

矩阵：

\[
EP_{j\rightarrow k}.
\]

真实 edge \(1\rightarrow2\) 应突出。

## Figure E：Seed stability

每个 seed 报告：

- test NRMSE；
- participation F1；
- mechanism functional \(R^2\)；
- relation edge F1 / EP；
- final PASS/FAIL。

---

# 42. Pass / Fail 验收标准

以下阈值是第一版工程验收标准，不是理论常数。

## Pass 0：Exact realizability

\[
\boxed{L_{oracle}<10^{-10}.}
\]

否则实验 invalid。

## Pass 1：Predictive recovery

建议：

\[
NRMSE_W<0.05
\]

在 test-IID 和 held-out combination 上均成立。

## Pass 2：Structural recovery

经过 permutation/gauge alignment：

\[
F1_m>0.9,
\]

\[
R^2_{func}>0.9.
\]

多余 candidates usage：

\[
u_{redundant}<0.1.
\]

## Pass 3：Compositional recovery

held-out \(101\)：

\[
NRMSE_{101}<0.1
\]

且不能明显差于 IID test。

## Pass 4：Relation recovery

E0-B：

\[
EP_{1\rightarrow2}>0
\]

且显著高于：

\[
EP_{2\rightarrow1}
\]

以及其它 false edges。

推荐：

\[
R^2_{relation}>0.8.
\]

## Pass 5：Optimization stability

至少：

\[
\boxed{8/10}
\]

optimization seeds 同时通过核心条件。

最终：

\[
\boxed{E0=PASS}
\]

只有当 Pass 0--5 同时满足。

---

# 43. 预期实验结果及其含义

| 观察到的结果 | 含义 | 下一步 |
|---|---|---|
| Oracle reconstruction 不接近 0 | generator 没有真正满足模型方程，或代码有 bug | 停止所有训练，修 generator |
| Oracle=0，但 learner train loss 很高 | 网络容量、optimizer、implementation 存在问题 | 简化 learner / 检查梯度 |
| Train loss 很低，但 test loss 高 | memorization / data coverage 不足 | 检查 split、capacity、regularization |
| Train/test prediction 都好，但 participation F1 很低 | objective 允许错误 latent decomposition；predictive fit 不等于 structure recovery | 研究 identifiability / participation pressure |
| Participation 好，但 mechanism functional recovery 差 | latent index 找对，但 realization/mechanism function 存在 gauge 或错误分解 | 检查 \(v\) alignment 与 mechanism parameterization |
| IID 好但 held-out \(101\) 差 | learner 没有发现 reusable compositional mechanisms | E0-A 不通过 |
| \(M_{\max}=5\) 时所有 candidates 高 usage | decomposition redundancy | 调 participation complexity / 检查 non-identifiability |
| No-\(R_D\) 在 relation world 上与 full model 一样好 | synthetic relation 太弱，或 individual mechanisms 吞掉 relation，或 \(R_D\) 没必要 | 检查 factorial design 与 staged training |
| Full \(R_D\) 降低 train loss但 held-out EP≈0 | relation overfitting | 不算 relation recovery |
| \(EP_{12}\) 和 \(EP_{21}\) 都高 | direction 未被辨识；可能是数据/参数化对称性 | 增加 directional contrasts，检查 relation definition |
| \(EP_{12}\gg EP_{21}\)，relation residual held-out recovery 高 | 正确恢复单向 relation | E0-B 通过 |
| 只有 1--2 个 seed recovery | optimization landscape 不稳定 | 不能宣称可可靠恢复 |
| \(\ge8/10\) seeds 结构与预测均恢复 | optimizer 在 realizable/evidence-sufficient setting 下具有稳定恢复能力 | 进入 E1/E2 |
| identity TP 成功但 orthogonal TP 失败 | learner 依赖 TP coordinate shortcut | E0 不应完全通过 |
| identity 与 orthogonal TP 都稳定成功 | recovery 不依赖原始 transformation coordinate | 强 E0 证据 |
| Negative controls 也全部“成功” | evaluation/data 设计无判别力或实现 bypass | 检查实验有效性 |

---

# 44. E0 成功后能够声称什么

如果 E0 通过，可以合理声称：

\[
\boxed{
\text{在一个严格属于所提出 hypothesis class 的 synthetic world 中，}
}
\]

\[
\boxed{
\text{并且 interaction data 提供充分机制组合与 relation contrast 时，}
}
\]

\[
\boxed{
\text{当前学习目标和优化实现能够稳定恢复 reusable mechanism decomposition 与 directed relation。}
}
\]

---

# 45. E0 成功后绝对不能声称什么

E0 不证明：

\[
\text{real world contains exactly these }Z_D,R_D.
\]

也不证明：

- 真实视觉可以形成正确 \(S\)；
- TP 能从 interaction 中自动学出来；
- 真实机器人 intervention 有足够 identifiability coverage；
- exploration 能找到缺失证据；
- task abstraction 有效；
- benchmark task success 具有竞争力。

E0 结论必须严格限制为：

\[
\boxed{
Realizable
+
Sufficient\ Evidence
\Rightarrow
Optimization\ Recovery.
}
\]

---

# 46. 推荐代码目录

```text
e0/
├── configs/
│   ├── e0a_identity.yaml
│   ├── e0a_orthogonal.yaml
│   └── e0b_relation.yaml
├── data/
│   ├── generator.py
│   ├── mechanisms.py
│   ├── relations.py
│   ├── tp_transform.py
│   └── splits.py
├── models/
│   ├── abstraction.py
│   ├── hard_concrete.py
│   ├── mechanism_bank.py
│   ├── relation_model.py
│   └── world_decoder.py
├── train/
│   ├── train_e0a.py
│   └── train_e0b.py
├── eval/
│   ├── oracle_certificate.py
│   ├── align_mechanisms.py
│   ├── metrics_prediction.py
│   ├── metrics_structure.py
│   ├── metrics_relation.py
│   └── plot_results.py
├── scripts/
│   ├── generate_data.sh
│   ├── run_e0a_all_seeds.sh
│   └── run_e0b_all_seeds.sh
└── outputs/
```

---

# 47. Dataset 文件结构

```text
datasets/e0_world_seed_20260901/
├── world_spec.json
├── ground_truth_world.pt
├── train/
│   ├── visible.npz
│   └── hidden_gt.npz
├── val/
│   ├── visible.npz
│   └── hidden_gt.npz
├── test_iid/
│   ├── visible.npz
│   └── hidden_gt.npz
├── test_context/
│   ├── visible.npz
│   └── hidden_gt.npz
└── test_combination_101/
    ├── visible.npz
    └── hidden_gt.npz
```

`visible.npz`：

```text
S
p
delta_S
```

`hidden_gt.npz`：

```text
m_gt
v_gt
e_gt
r_gt
```

训练代码禁止读取 `hidden_gt.npz`。

---

# 48. Generator 伪代码

```python
set_world_seed(20260901)

C = [
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
]

GT_params = initialize_fixed_mechanism_parameters(C)
B = random_orthogonal_matrix(dim=6)

for sample in dataset:

    S = sample_state()

    m = sample_balanced_participation(
        allowed_patterns=split_specific_patterns
    )

    v = sample_realizations(m)

    effects = []

    for j in range(3):
        e_j = zeros((3, 2))

        for k in range(3):
            if C[j][k] == 1 and m[j] == 1:
                e_j[k] = (
                    v[j] * A[j][k] @ S[k]
                    + v[j] * b[j][k]
                    + 0.25 * v[j]**2 * c[j][k]
                )

        effects.append(e_j)

    delta_S = sum(effects)

    if E0_B:
        relation = generate_relation_1_to_2(
            S=S,
            v1=v[0],
            v2=v[1],
            e2=effects[1],
            m1=m[0],
            m2=m[1],
        )
        delta_S += relation

    if tp_mode == "identity":
        p = flatten(delta_S)
    elif tp_mode == "orthogonal":
        p = B @ flatten(delta_S)

    save_visible(S, p, delta_S)
    save_hidden_gt(m, v, effects, relation)
```

---

# 49. 训练执行顺序

## Step 1：生成 world

```bash
python -m data.generator \
    --world-seed 20260901 \
    --mode e0a
```

## Step 2：Oracle certificate

```bash
python -m eval.oracle_certificate \
    --dataset datasets/e0_world_seed_20260901
```

只有 PASS 才继续。

## Step 3：E0-A Identity TP

```bash
python -m train.train_e0a \
    --config configs/e0a_identity.yaml \
    --seed 0
```

完成 10 seeds。

## Step 4：E0-A Orthogonal TP

```bash
python -m train.train_e0a \
    --config configs/e0a_orthogonal.yaml \
    --seed 0
```

完成 10 seeds。

## Step 5：结构评估

```bash
python -m eval.plot_results \
    --experiment outputs/e0a_orthogonal
```

## Step 6：只有 E0-A PASS 后生成 E0-B

```bash
python -m data.generator \
    --world-seed 20260901 \
    --mode e0b
```

## Step 7：训练 relation

```bash
python -m train.train_e0b \
    --config configs/e0b_relation.yaml \
    --seed 0
```

完成 10 seeds。

---

# 50. 必须保存的实验元数据

每个 run 至少保存：

```text
world seed
dataset seed
optimization seed
git commit hash
config
environment/package versions
training curves
best checkpoint
final checkpoint
all test metrics
mechanism alignment mapping
relation EP matrix
pass/fail decision
```

---

# 51. Reproducibility checklist

- [ ] ground-truth world 参数已固定并保存；
- [ ] world seed 与 optimizer seed 分离；
- [ ] learner 训练时未读取 hidden GT；
- [ ] oracle realizability certificate PASS；
- [ ] held-out \(101\) 未泄漏进 training；
- [ ] identity TP 与 orthogonal TP 均测试；
- [ ] structure metrics 使用 permutation alignment；
- [ ] realization comparison 允许合理 gauge alignment；
- [ ] 不仅报告 reconstruction loss；
- [ ] 至少 10 optimization seeds；
- [ ] negative controls 已执行；
- [ ] relation EP 使用 held-out data；
- [ ] \(g_{jk}\) 在 world equation 中只乘一次；
- [ ] 每个 run 保留完整 config；
- [ ] PASS/FAIL 根据预先设定标准，而非看结果后修改标准。

---

# 52. 最终 E0 报告建议结构

工程师完成实验后应提交：

1. **Experiment validity**：oracle error、dataset coverage、train/test split；
2. **E0-A identity TP**：prediction、participation、mechanism recovery、held-out composition、seed recovery rate；
3. **E0-A orthogonal TP**：同上；
4. **E0-B relation**：world prediction、edge gate、EP matrix、relation functional recovery、direction comparison；
5. **Negative controls**：逐项报告；
6. **Final judgement**：只能选择以下之一：

```text
E0 PASS
E0 PARTIAL PASS
E0 FAIL
E0 INVALID
```

并明确原因。

---

# 53. 核心结论判据

E0 最终不是回答：

> 模型 loss 能不能降下来？

而是回答：

\[
\boxed{
\text{当 ground-truth world 被人为保证确实由我们提出的结构产生时，}
}
\]

\[
\boxed{
\text{当 dataset 被人为保证具有足够的 mechanism/relation variation 时，}
}
\]

\[
\boxed{
\text{learner 是否能够在不知道 ground-truth ontology 的前提下，}
}
\]

\[
\boxed{
\text{通过当前 objective、network parameterization 和 optimization procedure，}
}
\]

\[
\boxed{
\text{稳定恢复具有正确 functional meaning 的 }Z_D\text{ 与 }R_D。
}
\]

只有 prediction、structure、composition、relation direction 和 seed stability 同时成立，E0 才真正完成它的任务。
