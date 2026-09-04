# E0 补充诊断实验总结：O5 / S1 / D1 / D2

> 本文件汇总本轮补做的四项诊断实验（O5、S1、D1、D2）的结果与结论。
> 所有实验均沿用 `WORLD_SEED=20260901`、固定 dataset、`outputs/e0_diagnostics/` 输出目录，
> 未覆盖 `outputs/e0a/main/` 的原始 E0-A 失败结果。
> 原始判定保持：**E0-A FAIL**。

---

## 1. O5 — Near-GT 初始化稳定性测试

**目的**：区分「GT 解存在但随机优化找不到」与「objective 本身把模型推离 GT 解」。

**设置**：用 O1 的 mechanism bank + O2 的 supervised q_eta 初始化，之后删除全部 GT supervision，
只用原始 `L_effect + λ_P L_participation`（λ_P=1e-3）继续训练 200 epochs。seed 0。

### 结果

| 配置 | init F1 | init inst-R² | final IID F1 | final inst-R² min | functional R² min | gt_stable |
|---|---:|---:|---:|---:|---:|:---:|
| O5-A m=3 identity | 0.984 | 0.973 | 0.984 | 0.990 | 0.976 | ✅ |
| O5-A m=3 orthogonal | 0.984 | 0.969 | 0.986 | 0.988 | 0.981 | ✅ |
| O5-B m=5 identity | 0.984 | 0.973 | 0.986 | 0.990 | 0.979 | ✅ |
| O5-B m=5 orthogonal | 0.984 | 0.969 | 0.980 | 0.988 | 0.982 | ✅ |

- **GT 结构在 200 epochs 无监督训练中长期保持**（F1≈0.98、instance-R²≈0.99、functional-R²≈0.98）。
- **O5-B（m=5）candidate 4/5 没有「重新打开」**：其 usage 在 epoch 10 内降至 0 并保持到结束，
  前 3 个正确 mechanism 全程保持 functional identity。
- `test_context` 上结构指标也保持（F1≈0.95–0.97）。
- `test_combination_101` 的 F1≈0.66、instance-R² 为极大负值是**指标退化假象**：
  101 split 恒为 m=[1,0,1]，Z2 从不参与，其 GT effect 恒为 0，导致 R² 分母为 0、Z2 的 F1 恒为 0。

### 实现说明（与文档 14.1 的偏差）

文档要求 candidate 4/5 初始 `E[m_4],E[m_5]≈0`。当前实现用 `gate_offsets[3:]=-8.0` 抑制，
但 O2 中未训练的随机 query（candidate 3/4）经训练好的 gate_head 产生的 logit 过大，-8 不足以压到 0：
实测 init usage[3:]=[0.60, 0.67]。**因此 O5-B 不是严格的「从 0 重新打开」测试**，
而是观察到反方向：冗余 candidate 从 0.6 被 sparsity objective 压到 0。
这同样指向「当 3 个 GT mechanism 就位时，objective 正确关闭冗余项」。

### 结论

GT decomposition 是一个**稳定的 basin**（near-GT init 不被推走）。随机初始化失败不是 objective
把 GT 推开，而是**随机优化落入了另一个（dense/distributed）basin**。

---

## 2. S1 — λ_P Sweep

**目的**：测试单纯改变 sparsity strength 是否存在「good prediction + correct participation + correct mechanism」同时成立的工作区间。

**设置**：identity TP、M_max=5、λ_P ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}，各 1 seed。

### 结果

| λ_P | IID NRMSE | 101 NRMSE | active count | F1 | instance-R² min | good working point |
|---:|---:|---:|---:|---:|---:|:---:|
| 1e-4 | 0.100 | 0.119 | 5.00 | 0.688 | −3.14 | ❌ |
| 1e-3 | 0.114 | 0.152 | 4.33 | 0.657 | −0.50 | ❌ |
| 1e-2 | 1.004 | 1.003 | 0.043 | 0.000 | −0.01 | ❌ |
| 1e-1 | 1.004 | 1.003 | 0.039 | 0.000 | −0.01 | ❌ |
| 1e0  | 1.004 | 1.003 | 0.025 | 0.000 | −0.01 | ❌ |

### 结论

**不存在工作区间。** 呈现文档 17.4 预言的「单调退化」：
- λ_P ≤ 1e-3：所有 candidate 全开（active≈4.3–5），dense decomposition，F1≈0.66–0.69，instance-R² 始终很低。
- λ_P ≥ 1e-2：sparsity 直接把模型压塌到「预测全 0」（NRMSE→1.0，active→0.04，F1=0）。

即 **sparsity 本身不能提供 mechanism identity**。这不是 regularization scale 没调对，而是 local sparsity
在当前 objective/parameterization 下无法区分「正确 3 个 mechanism」与「任意 5 个 distributed mechanism」。

---

## 3. D1 — Leave-One-Candidate-Out

**目的**：验证当前 failed M=5 解是否真的构成 distributed cooperative code（而非存在可抵消的冗余项）。

**设置**：冻结 failed checkpoint，`G_l = MSE(ΔS, Σ_{r≠l} ê_r) − MSE(ΔS, Σ ê_r)`。

### 结果

identity + orthogonal × seed 0/1/2（6 组）全部 `distributed_cooperative_code = True`：
每个 candidate 的 `G_l` 在 test_iid / test_context / test_combination_101 上均 > 0。

代表性数值（identity seed 0，test_iid，G_l）：

| candidate | E[m_l] | E‖e_l‖ | G_l |
|---|---:|---:|---:|
| 1 | 1.00 | 1.20 | 0.247 |
| 2 | 1.00 | 0.95 | 0.166 |
| 3 | 1.00 | 0.80 | 0.114 |
| 4 | 1.00 | 0.70 | 0.092 |
| 5 | 1.00 | 1.19 | 0.254 |

### 结论

**5 个 candidate 都已成为预测所必需的 distributed representation**，不存在「effect 很大但可被其它
candidate 抵消」的冗余项。删除任一 candidate 都会显著恶化预测。

---

## 4. D2 — Latent Probe

**目的**：判断 GT latent structure 是否仍以 distributed/mixed 形式编码在 `v̂_{1:5}` 中。

**设置**：冻结 failed model，用 train/test 上的 `v̂_{1:5}` 分别 probe GT participation 与 GT realization，
probe 用 linear 与 2-layer MLP 各一。identity + orthogonal × seed 0/1/2。

### 结果（6 组一致）

| probe task | linear | 2-layer MLP |
|---|---:|---:|
| participation F1 | ~0.53–0.57 | ~0.89–0.91 |
| realization R² | ~0.44–0.49 | ~0.61–0.63 |

### 结论

- GT **participation** 以**非线性 distributed 形式**部分编码在 learned latent 中（linear 低、MLP 高）。
- GT **realization** 值 `v*` 即使非线性 probe 也只能恢复到 R²≈0.62（<0.9），
  说明 learner 对 realization 已形成**基本不同的 predictive coordinates**。

注：probe 成功 ≠ mechanism discovery 成功；这里 probe 只用于理解失败形式。

---

## 5. 整合诊断结论（结合已完成的 O1–O4 / O3-E / O3-A）

| 实验 | 结果 | 含义 |
|---|---|:---|
| O1 GT m,v → learned mechanism | **PASS**（3 seeds） | mechanism network capacity 排除 |
| O2 supervised q_eta | **PASS**（identity + orthogonal） | q_eta inference capacity 排除 |
| O3-E exact identifiability | **PASS**（top-1 recovery 0.997，positive margin 0.997） | objective 本身可辨识：真实 participation 是全局最优 |
| O3-A GT mechanisms + unsup q_eta | **FAIL**（F1≈0.69） | 即便 mechanism bank 固定正确，随机初始化的 amortized inference 也找不到 GT 解 |
| O4 full model M_max=3 | **FAIL** | overcomplete bank（M=5）不是主因；order 正确仍失败 |
| O5 near-GT init | **PASS / STABLE** | GT 是稳定 basin，不被 objective 推走 |
| D1 leave-one-out | distributed cooperative code | 失败解是 5 个 candidate 的 dense 协作 |
| D2 latent probe | 非线性才部分可探 | GT factors 以 distributed 形式存在 |
| S1 λ_P sweep | 无工作区间 | local sparsity 无法提供 mechanism identity |

### 最终失败来源判定

```text
G. Random-initialization / symmetry-breaking failure
   （amortized inference 的优化 / basin-of-attraction 问题）
```

**理由链**：
1. O3-E 证明 objective（effect + sparsity）**可辨识**（排除 C）；
2. O1/O2 证明 mechanism 与 q_eta **容量足够**（排除 A/B）；
3. O4 证明 **overcomplete M=5 不是主因**（排除 E）；
4. O5 证明 GT 是**稳定 basin**、不被 objective 推离（排除 F）；
5. O3-A（随机 init 失败）+ O5（near-GT init 稳定）共同指向：**随机初始化落入错误的 dense/distributed
   basin，而 optimizer 无法从那里跃迁到 GT basin**。

即：问题不在「模型学不会」或「目标函数错」，而在**随机初始化下的 amortized inference 无法破坏对称性、
落到正确的稀疏 mechanism 分解上**。下一步不应继续调 λ_P 或网络，而应回到 identifiability principle，
考虑引入比 local sparsity 更强的机制身份信号（如 cross-interaction consistency、intervention provenance、
independent variation、temporal reuse 等）。
