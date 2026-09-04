# E0-ADP1 阶段性实验报告

> 实验：Exact Alternating Sparse Coding / EM-style alternating optimization  
> 状态：探索性报告（目前只有 optimization seed 0，不是完整多 seed 确证实验）  
> 日期：2026-09-03

## 0. 执行摘要

E0-ADP1 的目的不是证明 EM-style 优化具有全局收敛保证，而是检验一个更窄的问题：切断 assignment inference 与 mechanism learning 的同步 co-adaptation 后，随机初始化是否更容易进入 O5 已证明存在且稳定的 GT-like basin。

当前答案是否定的。在 guide 配置的 1,024 样本实验中，ADP1 虽然比 paired joint baseline 得到更低的预测误差，但没有恢复 GT ontology：IID NRMSE 为 0.3633，participation F1 为 0.5714，最小 matched instance-effect \(R^2\) 为 -0.2813，最小 functional \(R^2\) 为 -0.4020；五个 candidate 的 leave-one-out gain 全部为正，说明它们仍共同承担 reconstruction。连续轮次的 assignment 也没有稳定，per-bit support change rate（SCR）长期约为 0.20--0.24。

增加 M-step 强度不能修复该问题。将每个 candidate 每轮的更新从 16 步增至 200 步后，1,024 样本的 IID NRMSE 和 functional \(R^2\) 反而恶化。将训练集扩大到 4,096 和 16,384 后，IID NRMSE 改善到 0.339 和 0.304，但 functional recovery 仍为负，16,384 样本运行的最小 matched functional \(R^2\) 为 -0.3358，最终 SCR 为 0.2876。

因此，本阶段支持以下结论：

\[
\boxed{\text{仅靠 exact support enumeration、更多样本或更深 M-step，尚不能使 ADP1 进入 GT-like basin。}}
\]

这不是完整的 seed-stability 结论。目前只有 seed 0 具备完整的 ADP1/joint 指标，故形式上的 observed RecoveryRate 为 \(0/1\)，不能替代至少 5--10 个 seed 的正式估计。

---

## 1. 实验目的与待检验假设

原 joint optimization 同时更新：

\[
q_\eta(S,p)\longrightarrow (m,v),
\qquad
\mathcal M_j(S,v_j;\Theta_j),
\]

因此 candidate 可以在决定“由谁解释样本”的同时改变“自己解释什么”，形成 distributed code。ADP1 将两部分更新交替冻结：

1. E-step 冻结全部 mechanisms，精确枚举 support 并优化连续强度 \(v\)；
2. M-step 固定 \((m,v)\)，用 residual/backfitting 更新 mechanisms；
3. world discovery 阶段不训练 \(q_\eta\)，只有 assignment 稳定后才允许 amortization。

待检验假设为：如果 joint failure 主要由同步 co-adaptation 引起，那么 exact alternating 应当使多余 candidate 的 usage 随轮次下降，assignment 趋于稳定，并最终进入 O5 所显示的 GT-like basin。ADP1 不保证全局最优；本实验只检验上述具体优化路径是否有效。

---

## 2. 实验配置与控制变量

### 2.1 保持不变的项目

| 项目 | 设置 | 是否沿用 E0 |
|---|---:|---:|
| Synthetic world | E0A synthetic world | 是 |
| World seed | 20260901 | 是 |
| Dataset seed | 20260902 | 是 |
| Split | 原 train / val / test_iid / test_context / test_combination_101 | 是 |
| Subset selection | 每个 split 的 fixed prefix | 新增的资源约束，但不改变 world |
| \(M_{\max}\) | 5 | 是 |
| TP | identity | 是 |
| Mechanism bank | SetMechanismBank | 是 |
| Token dimension | 64 | 是 |
| Attention heads | 4 | 是 |
| Cross-attention layers | 2 | 是 |
| Participation penalty | \(\lambda_P=10^{-3}\) | 是 |
| 初始化 | optimization seed 0 的随机初始化；非 GT-like initialization | 与随机 joint 对照一致 |
| 额外模块/损失 | 无 relation、anchor curriculum、soft EM 或额外结构 loss | 按 ADP1 约束 |

Mini 主实验使用 train 1,024、val 256，以及每个 test split 512 个样本。配置文件列出 seeds 0、1、2，但本次实际完整产物仅有 seed 0。

### 2.2 ADP1 相对 joint baseline 的唯一算法改动

Joint baseline 令 inference network 与 mechanism bank 接受同一 reconstruction gradient，并通过 Hard-Concrete gate 联合学习 \((m,v,\Theta)\)。ADP1 保持数据、mechanism architecture、\(M_{\max}\) 和 \(\lambda_P\) 不变，只将联合梯度优化替换为冻结式 alternating optimization；world discovery 中移除 \(q_\eta\)。

#### E-step

对每个样本枚举 \(2^5=32\) 个 support，包括空 support。对给定 support \(m\)，只优化 active coordinates 的 \(v\)：

\[
v_j=1.5\tanh z_j,
\qquad
J_i(m,v)=L_{\mathrm{effect},i}(m,v)+\lambda_P|m|.
\]

使用 Adam，学习率 0.05；solver sanity 选择 rung A，即每个 support 20 个 inner steps、2 个 restarts。最终选择 \(J_i\) 最小的 support，平分时优先更稀疏的解释。实测连续目标 median \(|\Delta J|=0\)，未发现 E-step solver 错误。

#### M-step

M-step 不是普通 joint fit。固定 \((m,v)\) 后，对 candidate \(j\) 构造：

\[
r_i^{(j)}=\Delta S_i-\sum_{k\ne j}m_i^k\mathcal M_k(S_i,v_i^k),
\]

并只在 \(m_i^j=1\) 的样本上拟合

\[
\mathcal M_j(S_i,v_i^j)\approx r_i^{(j)}.
\]

优化器为 AdamW，学习率 \(3\times10^{-4}\)，weight decay \(10^{-5}\)，batch size 128，gradient clip 1.0。实现已经逐行核对 residual/backfitting 伪代码，未发现 M-step 实现错误。

这里采用固定 optimizer steps，而不是数据意义上的 epoch，因为各 candidate 的 active subset 大小不同：

| 运行 | 每 candidate/round | 每 round 总 candidate updates | Alternating rounds | 总 candidate updates |
|---|---:|---:|---:|---:|
| Mini guide | 16 steps | 80 | 12 | 960 |
| Strong-M diagnostic | 200 steps | 1,000 | 15 | 15,000 |
| Paired joint | 不适用 | 不适用 | 不适用 | 960 joint updates |

---

## 3. Pass/Fail 标准

单 seed PASS 要求所有主要条件同时成立：

| 分量 | 标准 |
|---|---:|
| Discovery stability | SCR \(<0.01\)，并满足稳定窗口/validation 条件 |
| IID prediction | NRMSE \(<0.05\) |
| 101 composition | NRMSE \(<0.10\) |
| Participation | mean F1 \(>0.90\) |
| Instance effect | min matched \(R^2>0.90\) |
| Functional recovery | min matched \(R^2>0.90\) |
| Redundancy suppression | max redundant usage \(<0.10\) |

这些阈值是预先规定的判断标准。不能因为本轮失败而用“loss 明显下降”替换 structure recovery 标准。

---

## 4. 每个 random seed 的最终结果

### 4.1 当前可用 seed

| Method | Seed | IID NRMSE | Context NRMSE | 101 NRMSE | F1 | mean AUROC | min instance \(R^2\) | min functional \(R^2\) | final IID usage | active count | PASS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| ADP1 | 0 | 0.3633 | 0.3785 | 0.4548 | 0.5714 | 0.5597 | -0.2813 | -0.4020 | [0.506, 0.529, 0.488, 0.506, 0.473] | 2.502 | FAIL |
| Joint | 0 | 0.5725 | 0.4809 | 0.7221 | 0.5764 | 0.6001 | -0.1644 | -0.0626 | [0.268, 0.785, 0.465, 0.053, 0.611] | 2.182 | FAIL |

ADP1 的三项 IID participation AUROC 分别为 0.5387、0.5803、0.5602；joint 分别为 0.6168、0.6899、0.4937。

### 4.2 RecoveryRate 与证据边界

现有产物只能计算 observed single-seed rate：

\[
RecoveryRate_{\mathrm{ADP1}}=\frac{0}{1}=0,
\qquad
RecoveryRate_{\mathrm{Joint}}=\frac{0}{1}=0.
\]

这不能称为 seed-stability 估计。配置中虽声明 seeds 0、1、2，但 seeds 1、2 未运行；也没有 5--10 个 seed 的最终表。正式结论至少需要补跑 seeds 1--4，理想设置为 seeds 0--9。鉴于当前 logger 缺少逐轮结构指标，应先补齐 telemetry，再投入多 seed 计算。

当前不存在 successful seed，因此不能提供“representative successful seed”曲线。下文给出唯一完整 failed seed 0，并明确所有未记录字段。

---

## 5. Alternating optimization 时间序列

### 5.1 Mini guide：seed 0 failed run

\(L_{\rm effect}\)、\(E[\#active]\)、validation penalized objective 和 SCR 已逐轮记录。逐轮 \(F1_m\)、instance \(R^2\)、validation NRMSE 以及每个 candidate 的 usage 没有记录，无法从最终 checkpoint 反推。

| Round | \(L_{\rm effect}\) | \(J_{\rm train}\) | \(E[\#active]\) | \(J_{\rm val}\) | SCR |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.09760 | 0.09945 | 1.846 | 0.09326 | — |
| 1 | 0.07903 | 0.08113 | 2.097 | — | 0.2201 |
| 2 | 0.06166 | 0.06398 | 2.315 | 0.06010 | 0.2340 |
| 3 | 0.05255 | 0.05493 | 2.378 | — | 0.2051 |
| 4 | 0.04479 | 0.04705 | 2.262 | 0.04391 | 0.2049 |
| 5 | 0.03817 | 0.04055 | 2.381 | — | 0.2328 |
| 6 | 0.03061 | 0.03304 | 2.430 | 0.03042 | 0.2215 |
| 7 | 0.02438 | 0.02686 | 2.481 | — | 0.2338 |
| 8 | 0.02030 | 0.02273 | 2.438 | 0.02186 | 0.2371 |
| 9 | 0.01721 | 0.01963 | 2.422 | — | 0.2287 |
| 10 | 0.01688 | 0.01929 | 2.417 | 0.01937 | 0.2369 |
| 11 | 0.01595 | 0.01836 | 2.407 | — | 0.1984 |

该轨迹没有呈现 \(5\rightarrow4\rightarrow3\) 的 candidate-level pruning 证据。它只显示每个样本的平均 active 数从约 1.85 上升并停在约 2.4；最终 population-level 五个 usage 全部约为 0.5。也就是说，每个样本只选择约 2--3 个 candidate，但整个数据集在五个 candidate 间持续分配责任，这与正确的三机制 ontology 不同。

### 5.2 16,384 样本 + strong M-step：seed 0 scale diagnostic

该诊断只记录 \(J_{\rm train}\)、SCR 和 active count。下表中的 \(L_{\rm effect}\) 由

\[
L_{\rm effect}=J_{\rm train}-10^{-3}E[\#active]
\]

推得；它不是额外训练日志。

| Round | \(L_{\rm effect}\) | \(J_{\rm train}\) | \(E[\#active]\) | SCR |
|---:|---:|---:|---:|---:|
| 0 | 0.09655 | 0.09842 | 1.864 | — |
| 1 | 0.05501 | 0.05714 | 2.134 | 0.2921 |
| 2 | 0.03050 | 0.03294 | 2.441 | 0.2814 |
| 3 | 0.02312 | 0.02558 | 2.461 | 0.2628 |
| 4 | 0.01893 | 0.02147 | 2.541 | 0.2689 |
| 5 | 0.01561 | 0.01815 | 2.541 | 0.2566 |
| 6 | 0.01393 | 0.01651 | 2.583 | 0.2587 |
| 7 | 0.01205 | 0.01462 | 2.562 | 0.2538 |
| 8 | 0.01069 | 0.01322 | 2.532 | 0.2546 |
| 9 | 0.00988 | 0.01239 | 2.519 | 0.2503 |
| 10 | 0.00923 | 0.01173 | 2.497 | 0.2494 |
| 11 | 0.00995 | 0.01243 | 2.486 | 0.2676 |
| 12 | 0.01045 | 0.01293 | 2.481 | 0.2703 |
| 13 | 0.01132 | 0.01374 | 2.429 | 0.2789 |
| 14 | 0.01196 | 0.01436 | 2.400 | 0.2876 |

Round 10 之后训练目标和 SCR 同时恶化。这不是“结构已稳定但尚未达到阈值”，而是 assignment 继续大幅重排并出现后期漂移。

---

## 6. Assignment stability

用户指定的 full-support agreement 为

\[
A^{(r)}=\frac1N\sum_i\mathbf 1[m_i^{(r)}=m_i^{(r-1)}].
\]

当前实现没有保存逐轮 \(m^{(r)}\)，也没有计算这个精确指标。实现记录的是 per-bit support change rate：

\[
SCR^{(r)}=\frac{1}{NM_{\max}}\sum_{i,j}
\mathbf 1[m_{ij}^{(r)}\ne m_{ij}^{(r-1)}].
\]

因此 \(1-SCR\) 只是 bit-level agreement，不能冒充 \(A^{(r)}\)。Mini seed 0 的 SCR 始终约 0.20--0.24；16,384 strong-M run 的 SCR 始终约 0.25--0.29。换言之，每一轮约有四分之一的 assignment bits 改变，ontology 没有逐渐冻结。

---

## 7. E-step support distribution

只有 mini seed 0 的最终 train assignment 被保存。其 cardinality distribution 为：

| Cardinality | Count | Fraction |
|---:|---:|---:|
| \(|m|=0\) | 96 | 9.38% |
| \(|m|=1\) | 116 | 11.33% |
| \(|m|=2\) | 275 | 26.86% |
| \(|m|=3\) | 329 | 32.13% |
| \(|m|=4\) | 166 | 16.21% |
| \(|m|=5\) | 42 | 4.10% |

最常见的 support patterns 为：

| Pattern | Count | Fraction |
|---|---:|---:|
| 00000 | 96 | 9.38% |
| 11010 | 49 | 4.79% |
| 11111 | 42 | 4.10% |
| 01111 | 37 | 3.61% |
| 00111 | 37 | 3.61% |
| 01011 | 36 | 3.52% |
| 01101 | 35 | 3.42% |
| 10111 | 35 | 3.42% |
| 11011 | 33 | 3.22% |
| 10110 | 33 | 3.22% |
| 11101 | 33 | 3.22% |
| 00100 | 32 | 3.13% |

最终 distribution 并未集中到三个固定 candidate 所形成的 GT-compatible supports。\(|m|=3\) 虽是最大 cardinality 类别，但这不等价于恢复三个 mechanism identity；五个 candidates 的边际 train usage 分别为 [0.481, 0.509, 0.482, 0.518, 0.478]。

逐轮 cardinality 和 pattern distribution 未保存，因此不能判断 E-step 是否随轮次单调偏向 sparse explanations。这是下一版 logger 必须补齐的字段。

---

## 8. Candidate specialization

### 8.1 最终 candidate 指标

| Candidate | IID usage | \(E\|e_j\|\) | LOO gain \(G_j\) | GT \(Z_1\) instance \(R^2\) | GT \(Z_2\) instance \(R^2\) | GT \(Z_3\) instance \(R^2\) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5059 | 0.2255 | 0.02378 | -0.3450 | -0.0316 | -0.3640 |
| 2 | 0.5293 | 0.2533 | 0.02452 | -0.3131 | -0.0089 | -0.4044 |
| 3 | 0.4883 | 0.2182 | 0.01975 | -0.2813 | 0.0371 | -0.2580 |
| 4 | 0.5059 | 0.2964 | 0.03130 | -0.3669 | -0.0470 | -0.6208 |
| 5 | 0.4727 | 0.2260 | 0.01900 | -0.2411 | 0.0863 | -0.2757 |

\(E\|e_j\|\) 是在最终 train assignment 上，将 gate 后的 candidate contribution 展平后取 \(L_2\) norm，再对全部样本平均。由于 mini runner 只保存了 best-round bank 和 last-round assignments，这一列混用了 best bank 与 final assignment，应视为诊断量，不作为预注册 PASS 指标。

所有 \(G_j>0\)。删除任何一个 candidate 都会提高 IID reconstruction MSE，因此没有 candidate 被真正淘汰。candidate 1 和 4 虽在 Hungarian matching 中被标为 redundant，其 usage 分别仍为 0.506 和 0.506，远高于 0.10 阈值。

### 8.2 Functional \(R^2\) matrix

| Candidate | GT \(Z_1\) | GT \(Z_2\) | GT \(Z_3\) |
|---:|---:|---:|---:|
| 1 | -0.7765 | -0.6532 | -0.6728 |
| 2 | -0.2336 | -0.1178 | -0.4191 |
| 3 | -0.3038 | -0.2414 | -0.3746 |
| 4 | -0.3798 | -0.2186 | -0.4202 |
| 5 | -0.4209 | -0.2577 | -0.4020 |

Hungarian matching 为 \(Z_1\to C_3\)、\(Z_2\to C_2\)、\(Z_3\to C_5\)。匹配后三项 functional \(R^2\) 为 [-0.3038, -0.1178, -0.4020]，不存在可解释的 specialization。

---

## 9. 与原 joint baseline 的直接对照

| Method | IID NRMSE | F1 | mean AUROC | min instance \(R^2\) | min functional \(R^2\) | active count | max redundant usage | RecoveryRate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Joint | 0.5725 | 0.5764 | 0.6001 | -0.1644 | -0.0626 | 2.182 | 0.611 | 0/1 |
| ADP1 | 0.3633 | 0.5714 | 0.5597 | -0.2813 | -0.4020 | 2.502 | 0.506 | 0/1 |

ADP1 明显降低 prediction NRMSE，但没有提高 participation、instance-effect matching、functional recovery 或 single-seed recovery。事实上，seed 0 上 joint 的两项最小 \(R^2\) 还高于 ADP1，尽管仍远低于 PASS 标准。

因此不能用“ADP1 loss 更低”宣称成功：

\[
\boxed{\text{ADP1 没有显著提高 structure recovery，也尚无多 seed stability 证据。}}
\]

---

## 10. 失败 seed 的现象分类

Seed 0 的主要失败类型是：

1. **Assignment 持续振荡**：SCR 长期约 0.2--0.3，没有接近 0.01；
2. **Distributed mechanism code**：五个 candidates 均有约 0.47--0.53 的 IID usage，且所有 leave-one-out gains 为正；
3. **Prediction 有改善但 structure 错**：ADP1 优于 joint 的 prediction NRMSE，但 F1 和两类 \(R^2\) 均失败；
4. **没有真正 pruning**：平均 active count 约 2.4 不能说明只剩三个固定 mechanisms，population-level 五个 candidates 都在工作；
5. **后期漂移**：16,384 strong-M run 在 round 10 后 \(J\) 与 SCR 同时反弹。

它不是“一直保持每个样本 5 个 active”，也不是“过早 prune 到 1--2 个”。更准确的描述是：每个样本使用 2--3 个 candidate，但 candidate identity 在样本和轮次之间持续重排。

---

## 11. 101 held-out combination 单独分析

在 101 split 中 GT participation 恒为 \((Z_1,Z_2,Z_3)=(1,0,1)\)。因此只对 active 的 \(Z_1,Z_3\) 计算 effect \(R^2\)，对 inactive 的 \(Z_2\) 报 false positive 与预测效应范数。

| 指标 | ADP1 seed 0 |
|---|---:|
| 101 NRMSE | 0.4548 |
| Exact pattern accuracy | 0.0957 |
| \(Z_1\) effect \(R^2\) | -0.0100 |
| \(Z_3\) effect \(R^2\) | 0.0075 |
| Inactive \(Z_2\) false-positive rate | 0.5840 |
| \(E\|\hat e_2\|\) | 0.2825 |

这里没有对恒为零的 GT \(e_2\) 计算 \(R^2\)。结果表明 101 失败不仅是总 prediction error 较高：inactive mechanism 出现了高频误激活，同时两个应当 active 的机制也没有恢复其真实 effect。

---

## 12. 样本量与 M-step 强度诊断

| Train N | M-step steps/candidate/round | IID NRMSE | min functional \(R^2\) | 结论 |
|---:|---:|---:|---:|---|
| 1,024 | 16 | 0.363 | -0.402 | Guide 配置，FAIL |
| 1,024 | 200 | 0.424 | -0.56 | 更深 M-step 反而恶化 |
| 4,096 | 200 | 0.339 | -0.32 | 有限改善，SCR 仍约 0.23 |
| 16,384 | 200 | 0.304 | -0.336 | NRMSE 改善，但结构没有改善 |

1,024-strong 和 4,096-strong 数字来自本轮执行者提供的诊断摘要；16,384 数字来自 `diag_scale.py` 的完整日志。16,384 的 matched functional \(R^2\) 为 [-0.0585, -0.2969, -0.3358]。

从 4,096 到 16,384，IID NRMSE 继续改善，但最差 functional \(R^2\) 没有改善，SCR 反而在末轮达到 0.2876。这排除了“只要在 8 GB 显卡允许范围内继续增大 N，就会自然进入 GT basin”这一简单解释。没有必要在不改变算法或 telemetry 的情况下盲目扩大到 32,768 或 100,000。

---

## 13. 计算代价与环境

| 运行 | Wall-clock | 平均每 alternating round | Peak/设备信息 |
|---|---:|---:|---|
| Mini ADP1, N=1,024, 12 rounds | 149.6 s | 12.5 s | 历史元数据记录 peak GPU allocation 150,190,080 bytes |
| Paired joint, 960 updates | 21.4 s | 不适用 | 同一 mini 任务 |
| Strong ADP1, N=16,384, 15 rounds | 约 5 h 45 min | 约 23 min（含最终评估摊销前近似） | 当前 Python 为 torch 2.6.0 CPU build，CUDA unavailable |

现有 logger 没有分开记录每轮 E-step 和 M-step 时间，因此不能可靠报告二者各自耗时。Mini 日志只记录每轮结束时的累计 wall time；16,384 diagnostic 只记录 round 结果。

特别需要说明环境偏差：生成 synthetic dataset 的 manifest 记录的是 torch 2.5.1+cu121 且 CUDA 可用；但执行 16,384 diagnostic 和本报告补充评估时，当前 `python` 返回 torch 2.6.0 且 `torch.cuda.is_available()==False`。因此本次 16,384 wall-clock 不能代表 RTX 3070 Ti 的 CUDA 性能，也不能用 `nvidia-smi` 的系统总显存占用推断脚本峰值显存。该偏差不改变已经产出的预测和结构指标，但会影响速度与可扩展性结论。正式复现前必须固定并记录解释器路径、torch build 和 device。

---

## 14. 数据质量与实现排查

本阶段已实测排除：

| 假设 | 证据 | 结论 |
|---|---|---|
| E-step solver 有 bug | 连续目标 median \(|\Delta J|=0\) | 排除 |
| M-step residual/backfitting 实现错误 | 已逐行核对伪代码 | 排除 |
| Oracle mechanism 无法表示 \(\Delta S\) | oracle max MSE = 0 | 排除 |
| M-step 预算过弱 | 200 steps/candidate 未恢复结构 | 排除为单一原因 |
| 1,024 样本太少 | 扩至 4,096 和 16,384 后结构仍失败 | 排除为单一原因 |

这些结果不证明所有可能的 alternating schedule 都失败；它们只说明当前 hard E-step + residual M-step 的实现和 schedule 没有解决 basin-entry 问题。

---

## 15. 缺失证据与下一版 logger 要求

本报告没有伪造以下未采集信息：

- seeds 1--9 的最终结果；
- representative successful seed（当前没有 successful seed）；
- 每轮 \(F1_m\)、instance-effect \(R^2\)、functional \(R^2\) 和 validation NRMSE；
- 每轮五个 candidate 的 usage trajectory；
- 每轮 cardinality distribution 和 top support patterns；
- 精确 full-support agreement \(A^{(r)}\)；
- 分离的 E-step/M-step wall-clock；
- 16,384 run 的完整 checkpoint、最终 usage 和 participation 指标。

在补跑多 seed 前，runner 应当每轮保存：

```text
round_metrics.jsonl
  round, L_effect_train, J_train, val_nrmse
  f1_m, instance_r2_matched, functional_r2_matched
  scr_bit, assignment_agreement_exact
  candidate_usage[5]
  cardinality_histogram[0..5]
  top_support_patterns
  e_step_seconds, m_step_seconds

checkpoints/
  round_000_bank.pt
  round_000_assignments.npz
  ...
```

Early stopping 或用户中断时也必须保存当前 bank/assignments 并执行最终评估，避免为了取得 functional \(R^2\) 被迫跑满所有 rounds。

---

## 16. 最终结论

ADP1 成功切断了显式的同步梯度更新，但没有切断更一般的 alternating co-adaptation：一次 E-step 可在当前随机 basis 上形成分布式 assignment，随后的强 M-step 又把 mechanisms 拟合到这些 assignment；下一轮 E-step 因 basis 已改变而重新分配 responsibility。其表现是 reconstruction 快速下降，而 support identity 长期振荡。

\[
\boxed{
\begin{aligned}
&\text{ADP1 seed 0：FAIL；}\\
&\text{增加 M-step 深度：不能修复；}\\
&\text{训练样本由 1,024 增至 16,384：改善 prediction，但不改善 ontology recovery；}\\
&\text{多 seed RecoveryRate：尚未测定。}
\end{aligned}}
\]

因此下一步不应继续单纯堆样本或 M-step updates。应先为 round-level structure dynamics 建立完整观测，再设计能够抑制 assignment 重排或控制 mechanism 漂移的最小干预，并继续坚持单变量对照。

---

## 17. 结果来源

- `configs/e0_adp1_mini.json`
- `outputs/e0_adp1_mini/seed_0/adp1/metrics.json`
- `outputs/e0_adp1_mini/seed_0/adp1/outer_curves.json`
- `outputs/e0_adp1_mini/seed_0/adp1/stability.json`
- `outputs/e0_adp1_mini/seed_0/adp1/final_assignments.npz`
- `outputs/e0_adp1_mini/seed_0/joint_mini/metrics.json`
- `outputs/e0_adp1_mini/seed_0/joint_mini/run_metadata.json`
- `outputs/e0_adp1_mini/seed0_run.log`
- `outputs/e0_adp1_mini/diag_scale_n16384_seed0.log`
- 执行者提供的 1,024-strong 与 4,096-strong 诊断摘要

