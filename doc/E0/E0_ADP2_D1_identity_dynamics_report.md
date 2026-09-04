# E0 ADP2-D1：A1 逐轮机制身份诊断报告

## 1. 实验目的与结论

本诊断完全复用 A1 已保存的 12 个 round checkpoint 与 assignment 文件，**没有重跑或继续训练 A1**。它回答三个问题：

1. candidate 的复用误差下降，是否伴随真实机制恢复变好；
2. 局部解释差距 \(\Delta_i\) 是否能预测 GT support correctness；
3. A1 是否曾短暂形成真实机制，随后又把机制打散。

结论是：

\[
\boxed{E^{reuse}\downarrow\ \text{没有伴随}\ R^2_{func}\uparrow；实际观察到的是二者同时下降。}
\]

12 轮、5 个 candidate 合并后的相关性为：

\[
\operatorname{Pearson}(E^{reuse},R^2_{best})=+0.7453,
\qquad
\operatorname{Spearman}=+0.7917.
\]

若“复用误差越低，真实机制越成熟”成立，应观察到显著**负相关**。现在的正相关表示：随着训练推进，candidate 的复用误差降低，但其最佳 GT functional \(R^2\) 也变得更低、更负。A1 学到的是越来越稳定、越来越可复用的错误碎片，而不是真实机制。

因此：

- A0 的 endpoint discriminator 仍然成立：成功的 O5 mechanism 与失败的 ADP1 mechanism 可以由 reuse error 分开；
- 但 reuse error 不能直接解释为沿 A1 轨迹的“机制成形进度”；
- 当前结果不支持把 reuse error 作为单调奖励直接加入下一版训练；
- A1 没有 candidate 真正接近过 GT，因此没有发现“先恢复、后打散”的证据。

## 2. 数据与计算口径

- 输入：`a1_seed_0/checkpoints/round_000_bank.pt` 至 `round_011_bank.pt`；
- assignment：对应的 `assignments/round_000.npz` 至 `round_011.npz`；
- 每轮配对口径：round-\(r\) 的 **post-M mechanism bank** 与已保存的 round-\(r\) **pre-M assignment**；
- reuse 计算时重新优化 active \(v\)，但不更新 mechanism 参数；
- functional \(R^2\) 使用 5000 个 synthetic probe；
- 每个 candidate 的表中 \(R^2\) 是它对三个 GT mechanism 的最大值：

\[
R^2_{j,r,\mathrm{best}}=\max_{k\in\{1,2,3\}}R^2_{func}(C_j,Z_k^*).
\]

- 完整的每轮 \(5\times3\) functional matching matrix 保存在 `final_metrics.json`；
- 总诊断耗时 163.24 秒。

上述 checkpoint/assignment 配对反映每轮 M-step 后的 mechanism identity。由于 assignment 来自该轮 M-step 前，它不是“重新完整执行一轮 E-step 后”的评估；因此绝对数值应按该口径解释，但跨轮趋势及 GT functional probe 不依赖重新训练。

## 3. 逐轮复用误差：\(12\times5\)

| Round | C1 | C2 | C3 | C4 | C5 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8710 | 0.8547 | 0.8508 | 0.8673 | 0.8748 |
| 1 | 0.8040 | 0.8185 | 0.7876 | 0.8082 | 0.8688 |
| 2 | 0.7375 | 0.7429 | 0.7398 | 0.7762 | 0.7987 |
| 3 | 0.6683 | 0.6595 | 0.6634 | 0.6918 | 0.7238 |
| 4 | 0.5732 | 0.5883 | 0.5508 | 0.5895 | 0.6402 |
| 5 | 0.5619 | 0.5513 | 0.5521 | 0.5616 | 0.5931 |
| 6 | 0.5433 | 0.5053 | 0.5331 | 0.4965 | 0.5770 |
| 7 | 0.4668 | 0.5227 | 0.4865 | 0.3682 | 0.4891 |
| 8 | 0.4559 | 0.4509 | 0.4018 | 0.3631 | 0.4405 |
| 9 | 0.4279 | 0.4249 | 0.3759 | 0.3530 | 0.4505 |
| 10 | 0.3832 | 0.4211 | 0.3896 | 0.3129 | 0.4022 |
| 11 | 0.3856 | 0.4100 | 0.3185 | 0.3133 | 0.3277 |

五个 candidate 的 reuse error 都出现明显下降，但 round 11 仍全部高于 A0 给出的 \(\tau_{reuse}=0.1495\)。因此 A0 gate 不会错误地把这些 candidate 判为已经成形；问题在于“下降趋势”本身并不等于向 GT 靠近。

## 4. 每轮 candidate 与真实机制的对应程度

下表报告每个 candidate 对三个 GT mechanism 的最佳 functional \(R^2\)。完整 \(5\times3\) matrix 见机器可读结果。

| Round | C1 | C2 | C3 | C4 | C5 |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.0303 | -0.0272 | -0.0873 | -0.0287 | -0.0576 |
| 1 | -0.0967 | -0.0587 | -0.1496 | -0.0320 | -0.0788 |
| 2 | -0.1349 | -0.0440 | -0.2337 | -0.0466 | -0.0968 |
| 3 | -0.1316 | -0.0291 | -0.3316 | -0.0659 | -0.0813 |
| 4 | -0.1748 | -0.0173 | -0.3778 | -0.0787 | -0.0550 |
| 5 | -0.2198 | -0.0910 | -0.3527 | -0.1023 | -0.1303 |
| 6 | -0.2358 | -0.0674 | -0.3532 | -0.1238 | -0.1313 |
| 7 | -0.2426 | -0.1073 | -0.5150 | -0.1378 | -0.2654 |
| 8 | -0.3149 | -0.1897 | -0.5430 | -0.2811 | -0.3091 |
| 9 | -0.3128 | -0.2573 | -0.6175 | -0.3674 | -0.3936 |
| 10 | -0.4503 | -0.2820 | -0.5205 | -0.3382 | -0.3769 |
| 11 | -0.3487 | -0.3163 | -0.6108 | -0.3617 | -0.4515 |

没有任何 candidate 在任何一轮达到 \(R^2>0.5\)，更没有达到 \(R^2>0.9\)。它们并非逐渐恢复 GT 后才发生漂移，而是从第一轮起就没有形成真实机制，并在后续 backfitting 中继续专门化为错误分解。

## 5. Reuse 与 GT recovery 的相关性

### 5.1 合并统计

| 统计量 | 数值 | 支持“reuse 下降意味着 GT 恢复”吗？ |
|---|---:|---|
| pooled Pearson | +0.7453 | 否；期望方向应为负 |
| pooled Spearman | +0.7917 | 否；期望方向应为负 |

### 5.2 每个 candidate

| Candidate | Pearson | Spearman | lag-1 Spearman \(E^{reuse}_{r}\) vs. \(R^2_{r+1}\) |
|---|---:|---:|---:|
| C1 | +0.9426 | +0.9860 | +0.9727 |
| C2 | +0.7741 | +0.8811 | +0.8909 |
| C3 | +0.9764 | +0.9790 | +0.9000 |
| C4 | +0.8780 | +0.9720 | +0.9727 |
| C5 | +0.8916 | +0.9021 | +0.9364 |

同轮和 lag-1 结果都与预期相反。由于 round 同时驱动两个量，这些相关系数不能解释成因果效应；但它们足以否定“在该 A1 轨迹中，reuse 下降是接近 GT 的可观测先导信号”。

## 6. 局部 objective gap \(\Delta_i\) 的诊断

以下统计合并全部 12 轮，共 \(12\times1024=12288\) 个 sample-round observation。

| \(\Delta_i\) bin | N | Strict support accuracy | 匹配三个 GT 后的 accuracy |
|---|---:|---:|---:|
| 0–0.01 | 1010 | 0.0574 | 0.1554 |
| 0.01–0.02 | 871 | 0.0471 | 0.1493 |
| 0.02–0.05 | 1893 | 0.0586 | 0.1638 |
| 0.05–0.10 | 2046 | 0.0459 | 0.1642 |
| 0.10–0.20 | 2135 | 0.0407 | 0.1504 |
| >0.20 | 4333 | 0.1962 | 0.2765 |

结果不是严格的“完全无关”：最高桶有一定 enrichment。但它仍不具备机制身份判据所需的可靠性：

- 即使 \(\Delta_i>0.20\)，strict support accuracy 只有 19.6%；
- 匹配掉 permutation、忽略两个 redundant candidate 后也只有 27.6%；
- 0–0.20 内没有稳定的单调关系；
- 当前 relative margin 的分母包含 \(|J_{best}|\)。当 \(J_{best}\approx0\) 时比值会爆炸；`>0.20` 桶的平均 margin 达 \(8.35\times10^8\)，说明该桶混有严重的数值尺度异常。

因此 \(\Delta_i\) 最多可保留为局部数值稳定性或粗筛指标，不能当作 mechanism identity evidence。若后续还要研究 margin，应同时报告 absolute gap，并对 relative gap 使用稳定分母、截断或 `log1p` 变换。

## 7. 是否存在“短暂恢复 GT 后又被打散”

| Candidate | 最佳 round | 峰值 best \(R^2\) | round 11 \(R^2\) | 峰值后下降 |
|---|---:|---:|---:|---:|
| C1 | 0 | -0.0303 | -0.3487 | 0.3184 |
| C2 | 4 | -0.0173 | -0.3163 | 0.2991 |
| C3 | 0 | -0.0873 | -0.6108 | 0.5234 |
| C4 | 0 | -0.0287 | -0.3617 | 0.3330 |
| C5 | 4 | -0.0550 | -0.4515 | 0.3965 |

所有峰值都仍为负值。严格地说，A1 中不存在 candidate “曾经成为真实机制后来被重新打散”；只能说若干 candidate 早期与 GT 的偏差较小，之后变得更差。

## 8. 对下一步设计的含义

本诊断排除了一个关键但过强的假设：

\[
\text{跨 context 可复用}\ \not\Rightarrow\ \text{GT mechanism identity}.
\]

错误碎片也能在当前 assignment/backfitting 闭环中形成稳定规律。因此不建议立刻把原始 \(E^{reuse}\) 作为单独的训练奖励。下一步如果继续研究 reuse，应先寻找能排除“稳定错误碎片”的额外、但仍最小化的可观测条件，例如比较跨 intervention/context 的反事实迁移，而不是只测同一错误 ontology 内部的一致性。

这不改变 ADP2-A1 的原结论：切断同步 co-adaptation 的 exact alternating 仍未从随机初始化进入 O5 已证明存在的 GT-like basin。D1 进一步说明，失败不是“正确机制一度出现后被训练破坏”，而是 mechanism identity 从未形成。

## 9. 输出文件

诊断输出目录：`E0/outputs/e0_adp2_compact/d1_identity_dynamics/`

- `final_metrics.json`：完整逐轮结果，包括每轮 \(5\times3\) functional \(R^2\) matrix、matching、margin bins 与 transient 信息；
- `reuse_12x5.csv`：逐轮复用误差；
- `best_functional_r2_12x5.csv`：逐轮每个 candidate 的最佳 GT functional \(R^2\)；
- `summary.md`：精简机器生成摘要。

复现实行脚本：`E0/adp2/diagnose_d1.py`。
