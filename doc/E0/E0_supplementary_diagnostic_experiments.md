# E0 补充诊断实验执行方案

> **文档定位**  
> 本文档用于指导工程师在当前 E0-A 失败后，按统一流程完成补充诊断实验。  
> 目标不是通过调参“把 E0 调成功”，而是定位失败究竟来自：
>
> \[
> \boxed{
> \text{mechanism network capacity}
> \;\;vs\;\;
> q_\eta\text{ inference capacity}
> \;\;vs\;\;
> \text{objective / identifiability}
> \;\;vs\;\;
> \text{joint optimization / symmetry breaking}
> }
> \]
>
> 当前正式 E0-A 结果保持不变：
>
> \[
> \boxed{\text{E0-A FAIL}}
> \]
>
> 所有补充实验均属于 **diagnostic experiments**，不得覆盖或替换原始失败结果。

---

# 1. 当前已知现象

当前 failed model 的 candidate statistics：

| candidate | \(E[m_j]\) | \(E|v_j|\) | \(E[v_j]\) | std(\(v_j\)) | \(E\|e_j\|\) | std(\(\|e_j\|\)) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.7593 | -0.7593 | 0.1442 | 1.2017 | 0.2260 |
| 2 | 0.9982 | 0.6927 | -0.6904 | 0.1715 | 0.9482 | 0.3119 |
| 3 | 1.0000 | 0.8860 | -0.8860 | 0.0978 | 0.7976 | 0.2241 |
| 4 | 1.0000 | 0.5563 | -0.5466 | 0.2421 | 0.7047 | 0.2538 |
| 5 | 1.0000 | 0.5173 | +0.4895 | 0.2914 | 1.1874 | 0.2958 |

这说明当前退化不是：

\[
\text{3 个正确 mechanism + 2 个无效 candidate}
\]

而更接近：

\[
\boxed{
\text{5 个 candidate 全部参与 reconstruction 的 distributed decomposition}
}
\]

因此补充实验必须回答：为什么 learner 更偏好这种 distributed solution。

---

# 2. 补充实验总览

按如下顺序执行：

| 编号 | 实验 | 核心问题 |
|---|---|---|
| O1 | GT \(m,v\) → learned \(\mathcal M_j\) | mechanism network 是否有足够表达能力？ |
| O2 | supervised \(q_\eta:(S,p)\to(m,v)\) | abstraction network 是否能从输入中解出真实 latent？ |
| O3-E | GT \(\mathcal M_j\) + exhaustive latent search | 当前 objective 本身是否选择 GT decomposition？ |
| O3-A | GT \(\mathcal M_j\) + unsupervised \(q_\eta\) | 若 objective 可辨识，amortized inference 能否找到它？ |
| O4 | full model，\(M_{\max}=3\) | overcomplete bank 是否是主要原因？ |
| O5 | near-GT initialization | GT solution 是稳定 basin 还是会被 objective 推走？ |
| D1 | leave-one-candidate-out | 当前 5 个 candidate 是否真的形成 distributed cooperative code？ |
| D2 | latent probe | GT factors 是否以 distributed form 隐藏在 learned latent 中？ |
| S1 | \(\lambda_P\) sweep | sparsity 是否存在 prediction 与 structure 同时正确的工作区间？ |

推荐执行顺序：

```text
O1
→ O2
→ O3-E
→ O3-A
→ O4
→ O5
→ D1 / D2
→ S1
```

在上述实验完成之前：

```text
不要运行 E0-B relation
不要进入 E1
不要增加视觉、噪声、Gaussian realization 或 nonlinear TP
```

---

# 3. 通用实验约束

所有实验必须遵守以下条件。

## 3.1 固定 ground-truth world

沿用原始 E0：

```text
WORLD_SEED = 20260901
```

不得重新生成 ground-truth world。

## 3.2 固定 dataset

沿用原始：

```text
train: 100,000
val: 20,000
test_iid: 20,000
test_context: 20,000
test_combination_101: 20,000
```

保持：

\[
101
\]

完全不出现在正式 training data。

## 3.3 区分三类 seed

必须分别记录：

```text
world_seed
dataset_seed
optimization_seed
```

不得混淆。

## 3.4 保留原 E0-A checkpoint

任何补实验不得覆盖：

```text
outputs/e0a_original_fail/
```

建议新建：

```text
outputs/e0_diagnostics/
```

---

# 4. O1：GT \(m,v\) → learned mechanism bank

## 4.1 实验目的

验证：

\[
\boxed{
\text{当前 }\mathcal M_j\text{ network architecture 是否能够表示 ground-truth mechanism function}
}
\]

该实验完全移除：

\[
q_\eta.
\]

若 O1 失败，则当前问题首先是 mechanism parameterization / optimization，而不是 latent identifiability。

---

## 4.2 实验设置

直接读取 hidden GT：

\[
S_i,\quad m_i^*,\quad v_i^*,\quad\Delta S_i.
\]

设置：

\[
M=3.
\]

预测：

\[
\hat e_i^j
=
m_i^{j*}\mathcal M_j(S_i,v_i^{j*}),
\]

\[
\boxed{
\hat{\Delta S}_i
=
\sum_{j=1}^{3}\hat e_i^j.
}
\]

仅训练：

\[
\mathcal M_1,\mathcal M_2,\mathcal M_3.
\]

不训练：

\[
q_\eta.
\]

不使用：

\[
L_{\mathrm{participation}}.
\]

---

## 4.3 Loss

\[
\boxed{
L_{O1}
=
D_S
\left(
\Delta S,
\sum_jm_j^*\mathcal M_j(S,v_j^*)
\right).
}
\]

第一版：

\[
D_S=\mathrm{MSE}.
\]

---

## 4.4 推荐训练配置

```text
optimizer: AdamW
learning rate: 3e-4
batch size: 512
max epochs: 200
gradient clip: 1.0
seeds: 3
```

---

## 4.5 关键指标及含义

| 指标 | 含义 |
|---|---|
| Train NRMSE | mechanism bank 是否能拟合训练数据 |
| IID NRMSE | 是否学习到可泛化 mechanism function |
| Context NRMSE | mechanism 是否对新 state/realization context 泛化 |
| 101 NRMSE | 三个独立 learned mechanisms 能否组合到未见 combination |

主要指标：

\[
NRMSE_W
=
\frac{
\sqrt{E\|\Delta S-\hat{\Delta S}\|^2}
}{
\sqrt{E\|\Delta S-E[\Delta S]\|^2}
}.
\]

---

## 4.6 预期结果

### 预期 PASS

\[
NRMSE_{\mathrm{IID}}<0.05,
\]

\[
NRMSE_{101}<0.05.
\]

若达到：

\[
<0.02
\]

视为 strong pass。

### 结果含义

| 结果 | 含义 |
|---|---|
| O1 PASS | mechanism network capacity 基本排除 |
| O1 FAIL | 当前 \(\mathcal M_j\) 架构/optimizer 连已知 mechanism 都拟合不了；暂停后续 identifiability 讨论 |

---

# 5. O2：Supervised \(q_\eta\) Capacity Test

## 5.1 实验目的

验证：

\[
\boxed{
q_\eta(S,p)
\text{ 是否有能力从正式输入中恢复真实 }m,v
}
\]

这里允许使用 GT supervision，因为实验只检查 inference capacity。

---

## 5.2 实验设置

输入：

\[
S,\quad p.
\]

输出：

\[
\hat m_1,\hat m_2,\hat m_3,
\]

\[
\hat v_1,\hat v_2,\hat v_3.
\]

设置：

\[
M=3.
\]

固定 query 顺序：

```text
query 1 ↔ GT Z1
query 2 ↔ GT Z2
query 3 ↔ GT Z3
```

该实验不测试 permutation invariance。

---

## 5.3 TP 条件

必须分别运行：

### O2-I

\[
p=\operatorname{vec}(\Delta S)
\]

### O2-O

\[
p=B_{\mathrm{orth}}\operatorname{vec}(\Delta S)
\]

---

## 5.4 Supervised loss

Participation：

\[
L_m
=
\frac13\sum_j
BCE(\hat m_j,m_j^*).
\]

Realization：

\[
L_v
=
\frac{
\sum_j
m_j^*
(\hat v_j-v_j^*)^2
}{
\sum_jm_j^*+\epsilon
}.
\]

总损失：

\[
\boxed{
L_{O2}=L_m+\lambda_vL_v
}
\]

第一版：

\[
\lambda_v=1.
\]

---

## 5.5 推荐配置

```text
optimizer: AdamW
learning rate: 3e-4
batch size: 512
max epochs: 100
seeds: 3
```

---

## 5.6 关键指标及含义

| 指标 | 含义 |
|---|---|
| Participation F1 | 是否正确判断 mechanism 是否参与 |
| Participation AUROC | participation ranking 是否具有辨识能力 |
| \(R_v^2\) | active mechanism realization 是否恢复 |
| Identity vs Orthogonal gap | q_eta 是否依赖 TP 原始坐标 |

Realization metric 只在：

\[
m_j^*=1
\]

的 sample 上计算。

---

## 5.7 预期结果

建议 PASS：

\[
F1_m>0.95,
\]

\[
AUROC_m>0.98,
\]

\[
R_v^2>0.95.
\]

### 结果含义

| 结果 | 含义 |
|---|---|
| identity + orthogonal 均 PASS | 输入信息与 q_eta capacity 基本排除 |
| identity PASS、orthogonal FAIL | q_eta 对 TP coordinate system 存在依赖 |
| 两者均 FAIL | abstraction architecture / input information path 有问题 |

---

# 6. O3-E：Exact Latent Identifiability Test

> **这是本轮最重要的实验。**

## 6.1 实验目的

直接回答：

\[
\boxed{
\text{当真实 }\mathcal M_j^*\text{ 已知时，}
L_{\mathrm{effect}}+\lambda_PL_0
\text{ 是否真的把真实 }(m^*,v^*)\text{ 作为最优解？}
}
\]

该实验尽量移除 neural optimization。

---

## 6.2 实验设置

固定：

\[
\mathcal M_j=\mathcal M_j^*.
\]

对每个 test sample：

\[
(S,\Delta S)
\]

枚举全部 participation：

\[
m\in\{0,1\}^3.
\]

共：

\[
8
\]

种。

对于每一个 candidate pattern \(\tilde m\)，求：

\[
\boxed{
J(\tilde m)
=
\min_{\tilde v}
D_S
\left(
\Delta S,
\sum_j
\tilde m_j\mathcal M_j^*(S,\tilde v_j)
\right)
+
\lambda_P\|\tilde m\|_0.
}
\]

最后：

\[
\hat m_{\mathrm{global}}
=
\arg\min_{\tilde m}J(\tilde m).
\]

比较：

\[
\hat m_{\mathrm{global}}
\stackrel{?}{=}m^*.
\]

---

# 7. O3-E 的两种 realization support

必须执行两个版本。

## 7.1 O3-E-Matched

active realization 限制在 GT support：

\[
v_j\in[-1,-0.3]\cup[0.3,1].
\]

因为区间不连续，需要枚举 sign。

最多 3 个 active variables：

\[
2^3=8
\]

种 sign assignments。

每种 sign 下对 magnitude 进行 bounded optimization。

推荐：

```text
optimizer: L-BFGS-B / scipy.optimize
samples: 1000
m patterns: 8
multiple v initializations: >=5
```

---

## 7.2 O3-E-Learner

使用正式 learner 实际允许的 \(v\) 范围。

例如若：

\[
v=\tanh(h)
\]

则：

\[
v\in[-1,1].
\]

允许：

\[
v\approx0.
\]

---

## 7.3 为什么需要两个版本

若：

\[
O3\text{-}E_{\mathrm{matched}}
\]

成功，而：

\[
O3\text{-}E_{\mathrm{learner}}
\]

失败，则说明 learner 的 realization domain 引入额外 gauge，例如：

\[
m=1,\ v\approx0
\]

可以替代：

\[
m=0.
\]

---

# 8. O3-E 关键指标

## 8.1 Participation top-1 recovery

\[
\boxed{
Acc_m
=
P(\hat m_{\mathrm{global}}=m^*)
}
\]

含义：

> 在全局搜索意义下，objective 是否选择真实 participation pattern。

---

## 8.2 Objective margin

GT objective：

\[
J_{\mathrm{GT}}
=
J(m^*).
\]

次优 alternative：

\[
J_{\mathrm{second}}.
\]

定义：

\[
\boxed{
\Delta J
=
J_{\mathrm{second}}-J_{\mathrm{GT}}.
}
\]

解释：

- \(\Delta J>0\)：GT 更优；
- \(\Delta J\approx0\)：近似不可辨；
- \(\Delta J<0\)：alternative decomposition 比 GT objective 更低。

---

## 8.3 输出表

| 指标 | Matched support | Learner support |
|---|---:|---:|
| GT participation top-1 recovery | | |
| median \(\Delta J\) | | |
| % \(\Delta J>0\) | | |
| % \(\Delta J\approx0\) | | |
| % alternative beats GT | | |
| active \(v\) MAE | | |

---

## 8.4 预期结果

理想情况：

\[
Acc_m>0.95
\]

且：

\[
P(\Delta J>0)>0.95.
\]

### 结果含义

| 结果 | 含义 |
|---|---|
| Matched PASS，Learner PASS | objective 在 fixed-GT-mechanism 条件下具有较强辨识能力 |
| Matched PASS，Learner FAIL | learner realization domain 引入额外 gauge |
| Matched FAIL | 当前 effect+sparsity objective 本身无法唯一支持真实 participation |
| 大量 \(\Delta J\approx0\) | 当前问题属于 non-identifiability，而不是 optimizer 单纯没训练好 |

---

# 9. O3-A：GT Mechanisms + Unsupervised \(q_\eta\)

## 9.1 实验目的

在：

\[
\mathcal M_j=\mathcal M_j^*
\]

已经固定正确的条件下，测试 amortized inference：

\[
q_\eta(S,p)
\]

是否能只通过原始 unsupervised objective 恢复 \(m,v\)。

---

## 9.2 设置

固定：

\[
M=3.
\]

冻结：

\[
\mathcal M_j^*.
\]

训练：

\[
q_\eta.
\]

Loss：

\[
\boxed{
L_{O3-A}
=
D_S
\left(
\Delta S,
\sum_j
m_j\mathcal M_j^*(S,v_j)
\right)
+
\lambda_PL_{\mathrm{participation}}.
}
\]

分别运行：

- identity TP；
- orthogonal TP。

每种 3 seeds。

---

## 9.3 关键指标

与正式 E0 相同：

- IID NRMSE；
- 101 NRMSE；
- participation F1；
- AUROC；
- instance effect recovery；
- active count。

---

## 9.4 预期结果

如果 O3-E PASS，理想 O3-A 也应接近：

\[
F1>0.9
\]

以及：

\[
NRMSE<0.05.
\]

---

## 9.5 结果含义

| O3-E | O3-A | 结论 |
|---|---|---|
| FAIL | FAIL | objective 本身不可辨 |
| PASS | FAIL | objective 有正确解，但 amortized inference / optimization 找不到 |
| PASS | PASS | inference 本身可行，full-model failure 来自 joint basis learning |
| FAIL | PASS | 优先检查 O3-E implementation |

---

# 10. O4：Full Model with \(M_{\max}=3\)

## 10.1 实验目的

测试：

\[
\boxed{
M_{\max}=5
}
\]

的 overcomplete candidate bank 是否是当前 dense decomposition 的主要原因。

---

## 10.2 设置

与正式 E0-A 完全一致，仅修改：

\[
\boxed{
M_{\max}=3.
}
\]

分别运行：

- identity；
- orthogonal。

每种：

```text
3 seeds
```

其它 hyperparameters 不变。

---

## 10.3 关键指标

- IID NRMSE；
- 101 NRMSE；
- participation F1；
- AUROC；
- instance-effect \(R^2\)；
- original functional \(R^2\)。

---

## 10.4 预期结果

### Case 1

若：

\[
M_{\max}=3
\]

后 structure recovery 明显改善：

\[
F1\rightarrow1,
\quad
R^2_{\mathrm{instance}}\rightarrow1,
\]

说明 overcomplete bank 是主要诱因之一。

### Case 2

若仍：

\[
F1\approx0.6\sim0.7
\]

且 mechanisms 仍不对应 GT，则说明即使 model order 正确，也存在：

\[
\boxed{
arbitrary mixed basis / factor identifiability problem.
}
\]

---

# 11. 新增指标：Instance Effect Recovery

## 11.1 目的

避免当前 functional \(R^2\) 对 realization coordinate gauge 过于敏感。

对每个真实 sample：

\[
e_i^{j*}
\]

已知。

learner 自己产生：

\[
\hat e_i^l
=
\hat m_i^l
\mathcal M_l(S_i,\hat v_i^l).
\]

定义：

\[
\boxed{
R^2_{\mathrm{instance}}(l,j)
=
R^2
\left(
\{\hat e_i^l\}_i,
\{e_i^{j*}\}_i
\right).
}
\]

形成：

\[
M_{\max}\times3
\]

matrix。

再进行 Hungarian matching。

---

## 11.2 指标含义

该指标不要求：

\[
\hat v_j=v_j^*
\]

也不要求：

\[
\mathcal M_j
\]

参数与 GT 相同。

只问：

> 在同一 interaction 上，learned candidate 实际贡献的 effect 是否等于 GT mechanism effect？

因此它是更接近理论中：

\[
Z_D^j
=
\text{reusable functional mechanism}
\]

的评价。

---

## 11.3 预期结果

strong structural recovery：

\[
\boxed{
R^2_{\mathrm{instance}}>0.9
}
\]

对 Hungarian matched 3 个 mechanisms 均成立。

---

# 12. O5：Near-GT Initialization Stability Test

## 12.1 实验目的

区分：

\[
\boxed{
\text{GT solution exists but random optimization cannot find it}
}
\]

和：

\[
\boxed{
\text{current objective itself drives the model away from GT solution}
}
\]

---

## 12.2 初始化来源

使用：

- O1 得到的 mechanism bank；
- O2 得到的 supervised \(q_\eta\)。

先确认初始化 structure metrics 接近 GT。

之后：

\[
\boxed{
删除所有 GT supervision。
}
\]

只使用原始：

\[
L_{\mathrm{effect}}
+
\lambda_PL_{\mathrm{participation}}.
\]

---

# 13. O5-A：\(M_{\max}=3\)

设置：

\[
M_{\max}=3.
\]

从 GT-like initialization 开始继续 unsupervised training。

记录：

- F1 随 epoch；
- instance \(R^2\) 随 epoch；
- NRMSE；
- expected active；
- \(v\) statistics。

---

# 14. O5-B：\(M_{\max}=5\)

前三个 candidate 使用 GT-like initialization。

candidate 4、5 初始化：

\[
E[m_4],E[m_5]\approx0.
\]

然后仅用 original unsupervised objective 训练。

---

## 14.1 关键观察

看：

\[
m_4,m_5
\]

是否再次主动打开。

并观察前三个正确 mechanisms 是否逐渐失去 functional identity。

---

## 14.2 预期结果及含义

| 结果 | 含义 |
|---|---|
| GT-like structure 长期保持 | GT decomposition 是稳定 local solution；随机初始化主要存在 symmetry-breaking / basin 问题 |
| \(M=3\) 保持，\(M=5\) 退化 | overcomplete model order selection 是关键问题 |
| \(M=3\) 也逐渐失去 GT identity | objective 本身对 GT decomposition 缺乏稳定约束 |
| candidate 4/5 从 near-zero 自动打开 | dense decomposition 在当前 objective 下具有优化优势 |

---

# 15. D1：Leave-One-Candidate-Out Analysis

## 15.1 实验目的

验证当前 failed \(M=5\) solution 是否真的形成：

\[
\boxed{
distributed cooperative code
}
\]

而不是存在 effect 很大但可以被其它 candidate 抵消的冗余项。

---

## 15.2 设置

使用当前已训练 failed checkpoint。

完整预测：

\[
\hat{\Delta S}
=
\sum_{l=1}^{5}\hat e_l.
\]

删除 candidate \(l\)：

\[
\hat{\Delta S}^{-l}
=
\sum_{r\neq l}\hat e_r.
\]

定义：

\[
\boxed{
G_l
=
D_S(
\Delta S,\hat{\Delta S}^{-l}
)
-
D_S(
\Delta S,\hat{\Delta S}
).
}
\]

---

## 15.3 输出

| candidate | \(E[m_l]\) | \(E\|e_l\|\) | \(G_l\) |
|---|---:|---:|---:|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

分别在：

- IID；
- context；
- 101

上计算。

---

## 15.4 指标含义

### 若五个 \(G_l\) 都显著 \(>0\)

说明：

\[
\boxed{
5\text{ 个 candidate 都已成为 prediction 所必需的 distributed representation}
}
\]

### 若某 candidate：

\[
E\|e_l\|
\]

很大但：

\[
G_l\approx0
\]

则说明可能存在：

\[
\boxed{
redundant / cancelling decomposition.
}
\]

---

# 16. D2：Latent Probe Diagnosis

## 16.1 实验目的

判断 GT latent structure 是否仍然被编码在：

\[
\hat v_{1:5}
\]

中，只是以 distributed / mixed form 存在。

---

## 16.2 数据

冻结 failed model。

对 train/test 保存：

\[
\hat v_i=
(\hat v_i^1,\dots,\hat v_i^5).
\]

不得更新原 model。

---

## 16.3 Probe A：GT participation

训练：

\[
\hat v_{1:5}
\rightarrow
m^*_{1:3}.
\]

分别使用：

1. linear logistic regression；
2. 2-layer MLP。

---

## 16.4 Probe B：GT realization

只对 active GT mechanism：

\[
\hat v_{1:5}
\rightarrow
v^*_{1:3}.
\]

分别：

1. linear regression；
2. 2-layer MLP。

---

## 16.5 指标

Participation：

- F1；
- AUROC。

Realization：

- \(R^2\)；
- MAE。

---

## 16.6 预期结果含义

| Probe 结果 | 含义 |
|---|---|
| linear probe 很高 | GT factors 可能只是 learned latent 的线性混合 |
| linear 低、MLP 高 | learned latent 是 nonlinear distributed reparameterization |
| linear/MLP 都低 | learner 形成了与 GT factorization 完全不同的 predictive coordinates |

注意：

\[
\boxed{
probe\ success \neq mechanism\ discovery\ success
}
\]

它只用于理解失败形式。

---

# 17. S1：\(\lambda_P\) Sweep

> **只有 O1--O5 完成之后再执行。**

## 17.1 实验目的

测试单纯改变 sparsity strength 是否存在：

\[
\boxed{
good\ prediction
+
correct\ participation
+
correct\ mechanism\ function
}
\]

同时成立的工作区间。

---

## 17.2 设置

第一轮只用：

\[
p=\Delta S.
\]

使用：

\[
M_{\max}=5.
\]

测试：

\[
\lambda_P
\in
\{
10^{-4},
10^{-3},
10^{-2},
10^{-1},
1
\}.
\]

每个：

```text
1 seed
```

若出现明显有希望的区间，再增加 seeds。

---

## 17.3 输出

| \(\lambda_P\) | IID NRMSE | 101 NRMSE | active count | F1 | instance \(R^2\) |
|---:|---:|---:|---:|---:|---:|

---

## 17.4 预期结果及含义

### 理想情况

存在某个 \(\lambda_P\)：

\[
NRMSE<0.05,
\]

\[
F1>0.9,
\]

\[
R^2_{\mathrm{instance}}>0.9.
\]

说明当前 objective 可能主要是 regularization scale 未调到正确区域。

### 如果只看到：

\[
\lambda_P\uparrow
\Rightarrow
active\ count\downarrow
\]

但：

\[
R^2_{\mathrm{instance}}
\]

始终很低，

则说明：

\[
\boxed{
sparsity\ 本身不能提供 mechanism identity。
}
\]

---

# 18. 补充实验推荐目录

```text
e0/
├── diagnostics/
│   ├── o1_gt_latent_mechanism_capacity.py
│   ├── o2_supervised_qeta.py
│   ├── o3e_exact_identifiability.py
│   ├── o3a_gt_mechanism_unsup_qeta.py
│   ├── o4_full_m3.py
│   ├── o5_near_gt_stability.py
│   ├── d1_leave_one_out.py
│   ├── d2_latent_probe.py
│   └── s1_lambda_sweep.py
│
├── configs/
│   └── diagnostics/
│
└── outputs/
    └── e0_diagnostics/
        ├── O1/
        ├── O2/
        ├── O3E/
        ├── O3A/
        ├── O4/
        ├── O5/
        ├── D1/
        ├── D2/
        └── S1/
```

---

# 19. 建议命令模板

具体参数名可按现有 repository 调整，但建议保持统一实验入口。

```bash
python diagnostics/o1_gt_latent_mechanism_capacity.py \
    --config configs/diagnostics/o1.yaml \
    --seed 0
```

```bash
python diagnostics/o2_supervised_qeta.py \
    --tp identity \
    --seed 0
```

```bash
python diagnostics/o2_supervised_qeta.py \
    --tp orthogonal \
    --seed 0
```

```bash
python diagnostics/o3e_exact_identifiability.py \
    --support matched \
    --num-samples 1000
```

```bash
python diagnostics/o3e_exact_identifiability.py \
    --support learner \
    --num-samples 1000
```

```bash
python diagnostics/o3a_gt_mechanism_unsup_qeta.py \
    --tp identity \
    --seed 0
```

```bash
python diagnostics/o4_full_m3.py \
    --tp identity \
    --m-max 3 \
    --seed 0
```

```bash
python diagnostics/o5_near_gt_stability.py \
    --m-max 5 \
    --seed 0
```

```bash
python diagnostics/d1_leave_one_out.py \
    --checkpoint outputs/e0a_original_fail/seed0.pt
```

```bash
python diagnostics/d2_latent_probe.py \
    --checkpoint outputs/e0a_original_fail/seed0.pt
```

---

# 20. 每个实验必须保存的结果

每个 diagnostic run 至少保存：

```text
config.yaml
metrics.json
training_curves.json
seed.json
checkpoint.pt   # 若有训练
stdout.log
```

O3-E 额外保存：

```text
samplewise_objective_table.csv
pattern_recovery.csv
margin_distribution.npy
```

O5 额外保存：

```text
structural_metrics_over_time.json
candidate_usage_over_time.json
```

---

# 21. 最终诊断决策矩阵

完成 O1--O5 后，根据下表判断失败来源。

| O1 | O2 | O3-E | O3-A | O4/O5 | 主要结论 |
|---|---|---|---|---|---|
| FAIL | — | — | — | — | mechanism architecture / optimizer failure |
| PASS | FAIL | — | — | — | q_eta inference capacity / input path failure |
| PASS | PASS | FAIL | FAIL | — | objective / identifiability failure |
| PASS | PASS | PASS | FAIL | — | amortized inference / optimization failure |
| PASS | PASS | PASS | PASS | M=3 PASS, M=5 FAIL | overcomplete model-order selection failure |
| PASS | PASS | PASS | PASS | random FAIL, near-GT PASS | optimization / symmetry breaking failure |
| PASS | PASS | PASS | PASS | near-GT 也逐渐退化 | objective 对 GT decomposition 不稳定 |
| PASS | PASS | PASS | PASS | full model PASS | 原 E0 失败主要属于训练/初始化工程问题 |

---

# 22. 预期最终结论模板

补实验完成后，最终报告不得只写：

```text
E0 supplementary experiments completed.
```

必须明确填入：

## 22.1 Mechanism capacity

```text
O1: PASS / FAIL
Evidence:
...
Interpretation:
...
```

## 22.2 Inference capacity

```text
O2: PASS / FAIL
Identity:
...
Orthogonal:
...
Interpretation:
...
```

## 22.3 Objective identifiability

```text
O3-E matched support:
...
O3-E learner support:
...
Median objective margin:
...
GT top-1 recovery:
...

Conclusion:
objective identifiable / weakly identifiable / non-identifiable
```

## 22.4 Amortized inference

```text
O3-A:
...
```

## 22.5 Joint learning

```text
O4:
...
O5:
...
```

## 22.6 Failure mode

最终必须从以下之一选择：

```text
A. Mechanism network capacity failure
B. q_eta inference failure
C. Objective non-identifiability
D. Amortized inference optimization failure
E. Overcomplete model-order selection failure
F. Joint-learning gauge degeneracy
G. Random-initialization / symmetry-breaking failure
H. Mixed failure mode
```

---

# 23. 本轮最关键的判断原则

不要以：

\[
L_{\mathrm{effect}}\downarrow
\]

作为补实验成功标准。

当前正式 E0 已经证明：

\[
\boxed{
good\ reconstruction
\not\Rightarrow
correct\ mechanism\ discovery.
}
\]

本轮真正要判断的是：

\[
\boxed{
\text{为什么 GT mechanism decomposition 没有成为 learner 的稳定、可辨识解。}
}
\]

其中 O3-E 是最关键的实验，因为它尽量移除了 neural optimizer：

\[
\boxed{
\text{如果真实 mechanisms 已知，当前 objective 是否在数学意义上偏好真实 participation？}
}
\]

如果答案是否定的，则下一步不应该继续调网络或学习率，而应回到：

\[
TP\rightarrow Z_D
\]

的 identifiability principle，重新讨论是否必须引入比 local sparsity 更强的：

- cross-interaction mechanism consistency；
- intervention provenance；
- independent intervention variation；
- mechanism equivalence under different actions；
- temporal reuse；
- 或其它能够真正定义 mechanism identity 的结构信号。

只有完成本轮诊断后，才决定是否修改数学建模。
