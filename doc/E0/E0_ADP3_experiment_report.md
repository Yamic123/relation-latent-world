# E0-ADP3 开发阶段实验报告

## 1. 执行结论

E0-ADP3 已按照执行方案的 stage gate 顺序运行至 Adp3-A1。

- **Adp3-0：PASS**；
- **Adp3-A1：FAIL**；
- 按方案“任何一步 FAIL 立即停止”的规定，**未执行 A2、B1、B2、C 和多 seed 正式确认**。

最重要的结果不是 fragment reconstruction 是否足够低，而是 clean single-change fragments 是否自动形成了三个真实 mechanism family。最终结果为：

\[
\boxed{
\text{一个 GT-like mechanism 明显形成，但另外两个仍被多个 candidate 混合分摊；A1 整体未通过。}
}
\]

最终 val fragment NRMSE 已降至 0.1459，但 assignment Hungarian accuracy 只有 0.5674，三个 matched functional \(R^2\) 为：

\[
[0.8886,\ 0.3232,\ -0.1760].
\]

这说明把学习对象从完整 TP 改成单一作用 fragment **确实比 Adp1/Adp2 更接近机制身份形成**，但仅靠 hard assignment + per-fragment latent \(v\) 的 mixture-of-mechanisms 仍不足以自动恢复全部三个真实 family。

## 2. 环境与配置

正式数值实验运行于：

```text
conda environment: wm
Python: 3.10.19
PyTorch: 2.5.1+cu121
device: CUDA
GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8 GB
optimization seed: 0
```

保持不变的科学条件：

```text
WORLD_SEED = 20260901
DATASET_SEED = 20260902
M_real = 3（仅 evaluator 可见）
M_max = 5
mechanism dim = 64
heads = 4
relation = none
TP = identity
```

Contrast dataset 使用执行方案原定完整规模：

| Split | Base groups | Contrasts/base | Fragments |
|---|---:|---:|---:|
| Train | 1024 | 12 | 12288 |
| Val | 256 | 12 | 3072 |
| Test | 512 | 12 | 6144 |

针对 8GB GPU 最终只保留了工程性调整：

```text
E-step chunk size = 256 fragments
M-step batch size = 128
```

单轮试跑的 PyTorch peak allocated memory 仅约 49 MiB，因此正式实验没有减少 base groups、contrast 数、网络宽度、inner steps 或 restarts。

A1 优化配置：

```text
alternating rounds = 40
inner optimizer = Adam
inner lr = 0.05
inner steps = 20
inner restarts = 2
v = 1.5 tanh(z)
M-step optimizer = AdamW
M-step lr = 3e-4
weight decay = 1e-5
M-step epochs/round = 4
functional probes = 2000
lambda_global = 0
merge/split = off
```

执行方案未规定 A1 总轮数。12-round pilot 时 NRMSE 和 assignment change 仍明显变化，因此保留 pilot 证据并将正式收敛预算扩到 40 轮。算法、数据和 PASS 阈值没有改变。

## 3. Adp3-0：contrast fragment 正确性

每个 base group 固定同一个 \(S_g\) 与 \((v_{g1},v_{g2},v_{g3})\)，生成全部 8 个 participation pattern，并从 12 条正向 single-change edge 提取：

\[
d_{ab}=\Delta S_b-\Delta S_a.
\]

Evaluator 对照真实单 mechanism effect 后得到：

| 指标 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| max fragment MSE | \(2.998\times10^{-15}\) | \(<10^{-10}\) | PASS |

Learner-facing visible 文件只包含：

```text
group_id, fragment_id, S, d
```

`y_gt`、`v_gt`、`e_gt` 和被切换的 GT index 仅存在于独立 evaluator 文件 `hidden_gt.npz`。

## 4. Adp3-A1 最终结果

### 4.1 PASS/FAIL 表

| 判据 | 最终结果 | 要求 | 判定 |
|---|---:|---:|---|
| Fragment Hungarian accuracy | 0.5674 | \(>0.90\) | FAIL |
| Matched functional \(R^2\) | [0.8886, 0.3232, -0.1760] | 每个 \(>0.80\) | FAIL |
| Min best-3 functional \(R^2\) | -0.1760 | \(>0.70\) | FAIL |
| Last-3 assignment change rate | 0.3197 | \(<0.10\) | FAIL |

因此：

\[
\boxed{\text{Adp3-A1 FAIL}}
\]

### 4.2 Reconstruction 与 assignment

| 指标 | Round 0 | Round 39 |
|---|---:|---:|
| Val fragment NRMSE | 0.7483 | 0.1459 |
| Fragment Hungarian accuracy | 0.3561 | 0.5674 |
| Assignment change rate | 1.0000 | 0.3171 |
| Restart assignment agreement | 约 0.73（首轮） | 0.6174 |

Reconstruction 明显改善，但 assignment 没有稳定。最后三轮仍约有 32% 的 train fragments 改变 candidate，说明 ontology 仍在持续重排。

### 4.3 Candidate usage

| Candidate | Final usage | Train sample count |
|---|---:|---:|
| C1 | 0.2100 | 2581 |
| C2 | 0.1405 | 1727 |
| C3 | 0.3116 | 3829 |
| C4 | 0.1099 | 1351 |
| C5 | 0.2279 | 2800 |

五个 candidate 都保持非平凡 usage；A1 没有产生两个自然低 usage candidate。

## 5. Mechanism identity 诊断

最终 functional \(R^2\) matrix（row 为 learned candidate，column 为 GT mechanism）为：

| Candidate | GT Z1 | GT Z2 | GT Z3 |
|---|---:|---:|---:|
| C1 | **0.8886** | -0.6672 | -0.2692 |
| C2 | -1.9102 | -1.0138 | -3.0287 |
| C3 | 0.3478 | -0.9359 | **-0.1760** |
| C4 | -1.6856 | -0.0507 | -1.0365 |
| C5 | -0.1099 | **0.3232** | -0.0035 |

Hungarian functional matching 为：

```text
GT Z1 -> C1, R² = 0.8886
GT Z2 -> C5, R² = 0.3232
GT Z3 -> C3, R² = -0.1760
```

C1 已形成较清楚的 Z1-like function。它在 round 20 后超过 0.90，并曾在 round 29 附近达到约 0.966，最终略降至 0.889。其余两个 GT mechanism 没有形成高质量 functional identity。

## 6. Fragment assignment 的实际分摊

最终 val assignment contingency（row 为 candidate，column 为 GT）为：

| Candidate | GT Z1 fragments | GT Z2 fragments | GT Z3 fragments |
|---|---:|---:|---:|
| C1 | 661 | 3 | 2 |
| C2 | 4 | 308 | 128 |
| C3 | 238 | 473 | 214 |
| C4 | 110 | 108 | 71 |
| C5 | 11 | 132 | 609 |

该表解释了 0.5674 的 clustering accuracy：

- C1 几乎成为纯 Z1 cluster；
- Z2 被 C2、C3、C4、C5 分摊；
- Z3 主要进入 C5，但也进入 C2/C3/C4；
- C3/C4 是明显的混合 cluster；
- assignment 上偏向 Z3 的 C5，在 functional probe 中却更接近 Z2，说明低 fragment error 不等价于全状态空间中的机制功能恢复。

## 7. 失败模式解释

本次 A1 不是以下失败：

```text
contrast generator 错误
GPU 预算不足
所有 candidate collapse 到单槽位
fragment reconstruction 完全学不动
```

实际失败模式是：

\[
\boxed{
\text{一个真实 family 成形} +
\text{其余 fragments 被多个仍不稳定的 candidate 按局部可拟合性分摊}
}
\]

每个 fragment 都带有自由优化的标量 \(v\)。即使 fragment 是纯的，不同 candidate 仍可通过 candidate function 与 per-fragment \(v\) 的共同重参数化，在局部样本上获得低误差。Hard assignment 会强化当前局部赢家，但没有明确约束同一 candidate 必须在整个状态空间内对应同一个 GT function family。

所以 clean fragment 解决了“完整 \(\Delta S\) 可被任意加法拆分”的问题，却没有完全解决：

\[
\text{mechanism function}\;\leftrightarrow\;\text{per-fragment realization }v
\]

之间的重新参数化与局部分区问题。

## 8. 为什么没有继续 A2

A2 的作用是合并已经形成的重复 function families，而不是修复尚未形成的 identity。当前仅一个 candidate 达到接近 GT 的 functional quality，另外四个并非简单的同机制复制品。

此时执行 merge sweep 可能把混合 candidate 合并成更少的错误模型，无法区分“数量恢复”与“身份恢复”。因此严格遵守方案：A1 FAIL 后停止。

## 9. 环境异常与等价处理

`conda wm` 中 NumPy/Matplotlib 的 Windows 原生线性代数 DLL 在 `svd`、`lstsq` 和绘图 transform 路径上出现 `0xc06d007f` 异常。处理方式为：

1. world 不重新通过 SVD 构造，直接读取原 E0 已序列化 world；这反而更严格保证 world 完全复用；
2. 标量 affine calibration 使用与 least squares 等价的闭式 covariance/variance 公式；
3. 所有训练和数值 evaluator 均在 `wm`+CUDA 中运行；
4. 最后的静态 PNG 绘图使用稳定的 base Python，仅读取 JSON，不参与训练或指标计算。

## 10. 计算代价

| 项目 | 实测 |
|---|---:|
| Formal A1 wall-clock | 414.8 s（约 6 分 55 秒） |
| 平均每轮 E-step（train+val） | 约 6.5–7.5 s |
| 平均每轮 M-step | 约 3.1–3.7 s |
| PyTorch peak allocated GPU memory | 48.94 MiB |

3070 Ti 8GB 对本开发实验有充足余量；本次失败不是由缩小模型或数据规模造成的。

## 11. 输出文件

根目录：`E0/outputs/e0_adp3/`

```text
adp3_0/final_metrics.json
contrast_data/visible_train.npz
contrast_data/visible_val.npz
contrast_data/visible_test.npz
contrast_data/hidden_gt.npz
a1_seed_0/checkpoints/round_000_bank.pt ... round_039_bank.pt
a1_seed_0/fragment_assignments/round_000.npz ... round_039.npz
a1_seed_0/round_metrics.jsonl
a1_seed_0/corrected_round_metrics.jsonl
a1_seed_0/corrected_evaluator_metrics.json
a1_seed_0/final_verdict.json
a1_seed_0/round_metrics.csv
a1_seed_0/a1_dynamics.png
a1_seed_0/a1_candidate_usage.png
a1_seed_0/summary.md
```

`final_verdict.json` 与 `corrected_round_metrics.jsonl` 是权威结果。原始 `round_metrics.jsonl` 中的 classification accuracy 曾误用 functional matching；训练本身不受影响，修正后的 assignment-contingency Hungarian 指标已从全部保存 checkpoint 只读重算。

## 12. 总结

Adp3 给出了比 Adp1/Adp2 更细致的结论：

\[
\boxed{
\text{单一作用 fragment 能显著降低分解难度，并允许至少一个真实 mechanism identity 成形；}
}
\]

但：

\[
\boxed{
\text{纯 fragment + hard mixture assignment 仍不足以可靠形成全部 }Z_D\text{ family。}
}

下一步不应直接进入全局数量合并，而应先针对 fragment family 的比较方式或 \(v\) 的跨样本约束做新的最小诊断。
