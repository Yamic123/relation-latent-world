# E0-ADP2 Compact（13 小时约束）阶段实验报告

> 日期：2026-09-03  
> 执行范围：ADP2-0 → ADP2-A0 → ADP2-A1  
> 最终状态：**A1 FAIL，按预注册阶段门控 STOP**

## 1. 13 小时约束下的规模调整

为保证单卡/单机环境中在 13 小时内产生可解释结论，本轮取消 16,384 中等规模、100k full dataset 和多 seed 确认，只保留 seed 0 的阶段因果链。基础 world、模型和目标保持不变：

| 项目 | 设置 |
|---|---:|
| World seed | 20260901 |
| Dataset seed | 20260902 |
| Context seed | 20260903 |
| Optimization seed | 0 |
| Train / val / tests | 1,024 / 256 / 每 split 512 |
| TP | identity |
| \(M_{\max}\) | 5 |
| Mechanism bank | SetMechanismBank, dim 64, heads 4, layers 2 |
| \(\lambda_P\) | \(10^{-3}\) |
| E-step | 32 supports，20 Adam steps，2 restarts |
| M-step | residual/backfitting，16 steps/candidate/round |

资源调整只有两项：E-step chunk 从 2,048 降到 256；ADP2-0 使用 4-round replay 验证新 logger，而已有 12-round ADP1 继续作为最终 baseline。chunk 改动不改变枚举语义。

由于运行时 `SonsOfTheForest.exe` 占用大部分 GPU 显存，CUDA 进程在首轮前被 WDDM 终止。本轮 compact development 随后固定使用 `E:\conda\python.exe``、torch 2.6.0 CPU build。所有 wall-clock 均不得与 CUDA 运行直接比较。

## 2. ADP2-0：日志补齐与基线复现

新 runner 每轮保存：bank checkpoint、assignment、bit SCR、exact support agreement、relative margin、resolved fraction、candidate usage/effect norm、support cardinality/top patterns、validation NRMSE，以及 E/M-step 时间。

与原 ADP1 前四轮比较，最大 train objective 绝对差为：

\[
0.004987<0.005.
\]

该差异落在预先声明的跨设备合理容差内，故 ADP2-0 compact **PASS**。

复现同时直接显示：resolved fraction 从 0.402 升到 0.583，但 SCR 仍约 0.21--0.24，exact full-support agreement 只有约 0.33--0.38。因此“大 margin”本身不等于跨轮稳定。

## 3. ADP2-A0：跨 context reuse 指标校准

使用 observable \(S\) 上两个固定正交随机投影与 train median，将样本划分为四个 context group。每个候选在所有四组均有至少 32 个 active samples，满足 evidence coverage 要求。

### 3.1 O5-B 正样本

| Candidate | Reuse error | Context coverage |
|---:|---:|---:|
| C1 | 0.01230 | 4/4 |
| C2 | 0.00887 | 4/4 |
| C3 | 0.01692 | 4/4 |

### 3.2 ADP1 failed checkpoint 负样本

| Candidate | Reuse error | Context coverage |
|---:|---:|---:|
| C1 | 0.34492 | 4/4 |
| C2 | 0.33144 | 4/4 |
| C3 | 0.43734 | 4/4 |
| C4 | 0.25074 | 4/4 |
| C5 | 0.40420 | 4/4 |

预注册判据：

\[
Q_{90}(E_{O5})=0.01599
<
Q_{10}(E_{ADP1})=0.28302.
\]

两组清晰分离，A0 **PASS**，并固定：

\[
\tau_{reuse}=0.14951.
\]

这支持“当前 reuse metric 能区分 GT-like mechanisms 与 ADP1 fragments”，但尚不证明把它加入训练就能形成 identity。

## 4. ADP2-A1：仅暂缓低证据 assignment

A1 相对 ADP1 的唯一训练改动为：E-step 仍选择 exact top-1 support，但只有 relative margin \(\Delta_i\ge0.05\) 的 resolved samples 可以进入 residual M-step。未加入 switch threshold、reuse gating、global cost、merge 或 split。

### 4.1 Round-by-round dynamics

| Round | Train objective | Val NRMSE | SCR | Exact agreement | Resolved | Candidate usage |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.09930 | 0.7945 | — | — | 0.402 | [.351,.347,.341,.438,.383] |
| 1 | 0.08681 | 0.7428 | 0.1686 | 0.4609 | 0.452 | [.320,.332,.393,.447,.379] |
| 2 | 0.07461 | 0.6899 | 0.1602 | 0.4863 | 0.542 | [.375,.370,.376,.438,.363] |
| 3 | 0.06392 | 0.6339 | 0.1523 | 0.5146 | 0.629 | [.353,.374,.386,.431,.381] |
| 4 | 0.05455 | 0.5790 | 0.1514 | 0.5449 | 0.680 | [.302,.391,.410,.420,.432] |
| 5 | 0.04582 | 0.5269 | 0.1783 | 0.5039 | 0.699 | [.383,.438,.488,.379,.435] |
| 6 | 0.03938 | 0.4502 | 0.1941 | 0.4746 | 0.760 | [.391,.475,.477,.436,.467] |
| 7 | 0.03007 | 0.4307 | 0.2250 | 0.4062 | 0.797 | [.481,.450,.510,.507,.413] |
| 8 | 0.02813 | 0.4119 | 0.2332 | 0.3975 | 0.819 | [.512,.463,.477,.473,.441] |
| 9 | 0.02476 | 0.3950 | 0.1906 | 0.5078 | 0.851 | [.488,.447,.454,.513,.391] |
| 10 | 0.02345 | 0.3773 | 0.1977 | 0.4990 | 0.831 | [.513,.410,.447,.481,.381] |
| 11 | 0.02167 | 0.3820 | 0.2012 | 0.5117 | 0.853 | [.508,.417,.446,.490,.395] |

A1 在早期把 SCR 降到约 0.15，但 round 5 后重新上升。resolved fraction 越来越高，并没有带来 ontology 稳定；五个 candidates 最终仍全部获得高 population usage。

### 4.2 最终结构指标

| 指标 | A1 seed 0 | A1 阶段要求 |
|---|---:|---:|
| Last-3 mean SCR | 0.1965 | 相对 ADP1 至少下降 30%，即约 <0.155 |
| IID NRMSE | 0.3754 | 观察项 |
| 101 NRMSE | 0.4187 | 观察项 |
| Participation F1 | 0.5744 | 观察项 |
| Matched instance \(R^2\) | [-0.189, 0.039, -0.184] | 应开始改善 |
| Matched functional \(R^2\), 5,000 probes | [-0.515, -0.317, -0.399] | min 相对 -0.402 提高至少 0.30 |
| Final candidate usage | [.430,.488,.512,.531,.398] | 不应五个均高 usage |

末三轮 SCR 只比 ADP1 的约 0.2214 下降 11%，functional minimum 约 -0.515，反而低于 ADP1 的 -0.402。因此 A1 **FAIL**。

## 5. 阶段 STOP 决策

ADP2 总路线要求任何一步 FAIL 都先解释，不能直接跳入更复杂版本。A1 的结果说明：

\[
\boxed{\text{仅阻止低-margin pseudo-assignments 参与 M-step，不足以形成正确 mechanism identity。}}
\]

它在早期减少了切换，但随后 resolved samples 自己仍形成 distributed code，assignment oscillation 回归，functional recovery 没有改善。这不是“只差更多轮数”：后半程 resolved fraction 已超过 0.8，而 SCR 和 candidate-wide usage 同时处于平台。

因此本次执行在 A1 后停止，未运行 A2、A3、B0、B1。继续 A2 会回答一个不同问题——切换门槛能否冻结当前错误分法——必须先针对 A1 failure 更新实验假设或由研究者明确授权，不能在本报告中自动视为原路线的成功延续。

## 6. 计算代价

| Stage | Wall-clock |
|---|---:|
| ADP2-0 4-round CPU replay | 372.8 s |
| ADP2-A0 reuse calibration | 21.6 s |
| ADP2-A1 12 rounds + final evaluation | 1,119.9 s |
| 5,000-probe functional recheck | 约 1 min |

纯实验计算约 26 分钟，远低于 13 小时上限；剩余预算被保留，因为预注册 STOP 已触发，而不是用于盲目增加复杂度。

## 7. 产物

```text
E0/configs/e0_adp2_compact.json
E0/adp2/run_compact.py
E0/outputs/e0_adp2_compact/validation.json
E0/outputs/e0_adp2_compact/adp2_0_seed_0/
E0/outputs/e0_adp2_compact/a0_reuse_calibration/
E0/outputs/e0_adp2_compact/a1_seed_0/
```

`a1_seed_0/functional_recheck_5000.json` 保存了与原 E0 相同 5,000 probes 的复核结果。每轮 checkpoint 和 assignment 均已保存，可用于后续诊断而无需重跑。

## 8. 最终结论

本次 compact 实验给出两个清晰、彼此独立的结论：

1. **A0 PASS**：跨 context reuse error 本身具有很强的正负 checkpoint 辨别力；
2. **A1 FAIL**：仅按 assignment margin 筛除低证据样本，不能使随机 mechanisms 形成 GT-like identity。

这意味着 reuse evidence 仍值得研究，但 margin filtering 不是足够的前置 identity-formation 机制。下一步应围绕“为什么高-margin resolved assignments 仍是错误 fragments”做诊断，而不是把 A1 的短暂 SCR 改善解释成成功。
