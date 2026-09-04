# E0-ADP5 开发阶段实验报告

## 1. 执行结论

实验按照 `Adp5_experiment_execution_plan.md` 的 stage gate 执行：

| Stage | 结果 | 后续动作 |
|---|---|---|
| Adp5-0：数据与泄露审计 | PASS | 进入 A0 |
| Adp5-A0：response family 完整性 | PASS | 进入 A1 |
| Adp5-A1：family-level mechanism discovery | **FAIL** | **STOP** |
| A2/A3/B/C/F | 未执行 | A1 gate 禁止继续 |

ADP5 得到的是一个有区分度的部分结果：

\[
\boxed{\text{把正负与多幅度 realization 放进同一 response family，确实消除了 ADP4 的 sign split。}}
\]

但：

\[
\boxed{\text{共享 candidate assignment 并没有使 family 内各点形成一致的 latent realization 坐标。}}
\]

最终 family-level sign-split 仅 0.0378，但三个 matched functional \(R^2\) 全部为负：

\[
[-0.4806,\ -0.4876,\ -0.6419].
\]

因此 ADP5 解决了“正负 realization 被分给不同 candidate”的外部离散 fragmentation，却没有解决 candidate function 与每点自由 latent \(v_r\) 之间的内部重参数化问题。

## 2. 实验配置

```text
conda env = wm
device = CUDA
GPU = NVIDIA RTX 3070 Ti Laptop, 8 GB
WORLD_SEED = 20260901
DATASET_SEED = 20260902
family seed = 20260905
optimization seed = 0
Mmax = 5
SetMechanismBank dim = 64, heads = 4
R_D = 0
```

正式实验没有缩减科学规模：

```text
train/val/test bases = 1024/256/512
K_g ∈ {1,2}
train amplitudes = [-1.0,-0.7,-0.4,0.4,0.7,1.0]
heldout amplitudes = [-0.85,-0.55,-0.3,0.3,0.55,0.85]
inner Adam lr = 0.05, 20 steps, 2 restarts
M-step AdamW lr = 3e-4, weight decay = 1e-5
batch = 32 families
epochs/round = 4
rounds = 40
```

只采用 256 response-point E-step chunk 作为显存控制。

## 3. Adp5-0：数据与泄露审计

| Split | Bases | Families | K=1 | K=2 | K=3 | GT counts | max/min |
|---|---:|---:|---:|---:|---:|---|---:|
| Train | 1024 | 1521 | 527 | 497 | 0 | [489,530,502] | 1.0838 |
| Val | 256 | 386 | 126 | 130 | 0 | [129,133,124] | 1.0726 |
| Test | 512 | 766 | 258 | 254 | 0 | [255,252,259] | 1.0278 |

所有 split 均满足：

- 每个 base 只展示 1 或 2 个 family；
- 没有 base 同时展示全部三个真实 mechanism；
- train GT coverage 的 max/min 小于 1.15；
- local family ID 在每个 base 内随机重排；
- learner-facing schema 不含 GT family、GT amplitude、GT \(m\) 或 \(M_{real}\)。

因此 Adp5-0 PASS。

## 4. Adp5-A0：response family 完整性

Train/val/test 的所有 family 均满足：

```text
family purity = 1.0
同时包含正、负 active realization = 100%
覆盖至少 3 个 |v| 区域 = 100%
6 train active + 6 heldout active + 1 baseline
```

Adp5-A0 PASS。

## 5. Adp5-A1 最终判定

| 判据 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| Family Hungarian accuracy | 0.7047 | >0.80 | FAIL |
| Matched functional \(R^2\) | [-0.4806,-0.4876,-0.6419] | 每个 >0.70 | FAIL |
| \(R^2>0.85\) 数量 | 0 | 至少 2 | FAIL |
| Last-3 assignment change | 0.3162 | <0.15 | FAIL |
| Mean GT fragmentation | 0.3004 | <0.30 | FAIL |

辅助指标：

```text
final train family MSE = 0.001642
final heldout-realization NRMSE = 0.183116
final family exact restart agreement = 0.67324
final mean sign-split = 0.03782
final mean magnitude-split = 0.04409
wall-clock = 853.37 s（约 14 分 13 秒）
PyTorch peak allocated GPU memory = 48.93 MiB
```

## 6. 训练过程中的短暂结构恢复

Family accuracy 在 round 6 曾达到：

\[
0.8212,
\]

同时 mean GT fragmentation 降到：

\[
0.1822.
\]

这两个 assignment-level 指标短暂满足 A1 要求。然而同轮 functional recovery 仍然失败；整个 40-round 轨迹中，任一 matched functional \(R^2\) 的最高值只有 0.1795。随后 assignment accuracy 回落且持续振荡。

所以不能把 round 6 解释为真正进入 GT-like mechanism basin。它只形成了较正确的 family 分类，没有形成正确的机制函数与 realization parameterization。

## 7. 最终 family contingency

Val contingency（row=candidate，column=GT family）：

| Candidate | GT Z1 | GT Z2 | GT Z3 |
|---|---:|---:|---:|
| C1 | 0 | 0 | 68 |
| C2 | 71 | 0 | 0 |
| C3 | 0 | 133 | 0 |
| C4 | 0 | 0 | 56 |
| C5 | 58 | 0 | 0 |

所有 candidate 的 GT purity 实际上都是 1.0：

- Z2 已整体进入 C3；
- Z1 被 C2/C5 两个纯 candidate 分摊；
- Z3 被 C1/C4 两个纯 candidate 分摊。

这与 ADP4 的混合分配不同。ADP5 已经让 candidate 获得了 family label purity，但仍保留同一 GT family 的重复 candidate，而且这些重复并非由正负 realization 造成。

最终 train candidate usage 为：

\[
[0.1650,\ 0.1847,\ 0.3485,\ 0.1650,\ 0.1368].
\]

## 8. 为什么 functional recovery 仍然失败

每个 response family 共享 candidate，但六个 response point 仍分别自由优化自己的 latent：

\[
v_{U,1},\ldots,v_{U,6}.
\]

只要求同一 candidate 重建六个 response，并不要求 inferred latent 随真实 intervention strength 保持统一的方向、排序或跨 state 对齐。

最终只读 latent 诊断显示：

| Candidate | 对应 GT | pooled corr(true amplitude, inferred latent) | mean within-family corr | 负方向 family 比例 |
|---|---|---:|---:|---:|
| C1 | Z3 | -0.386 | -0.494 | 0.853 |
| C2 | Z1 | -0.143 | -0.170 | 0.606 |
| C3 | Z2 | -0.318 | -0.400 | 0.789 |
| C4 | Z3 | -0.343 | -0.413 | 0.786 |
| C5 | Z1 | -0.206 | -0.259 | 0.621 |

同一真实 amplitude 在不同 state 下对应的 inferred latent 方差也很大。这说明模型没有形成一个跨 family 一致的 realization coordinate。它可以使用不稳定甚至非单调的 latent 编码拟合离散 response points，因此 heldout reconstruction 尚可，但在随机状态与连续 probe 上无法恢复真实 mechanism function。

## 9. 与 ADP4 的对照

ADP5 的 sign-split 是按 family 的 negative/positive-dominant observable response 定义；ADP4 指标直接按单点 GT \(v\) 正负定义，数值口径不完全相同，但方向仍具有诊断意义。

| 指标 | Adp4-A1 | Adp5-A1 |
|---|---:|---:|
| Validation/heldout NRMSE | 0.1045 | 0.1831 |
| Hungarian accuracy | 0.5404 | **0.7047** |
| Matched functional \(R^2\) | [0.3595,0.1608,0.1704] | [-0.4806,-0.4876,-0.6419] |
| Assignment change | **0.1068** | 0.3162 |
| Sign-split | 1.0000 | **0.0378** |
| Mean GT fragmentation | 0.4596 | **0.3004** |

ADP5 明显改善了 family purity、fragmentation 和 realization-sign invariance，但牺牲了稳定性，且没有形成正确 functional identity。

## 10. 为什么不进入 A2

A2 要验证已经形成的 candidate 能否跨 base 复用。当前 learner-visible reconstruction error 与 evaluator functional identity 已明显脱钩，而且所有 matched functional \(R^2\) 都为负。继续计算 reuse score 无法满足“A1 有明显正结果”这一前置条件。

按执行方案：

\[
\boxed{\text{A1 FAIL}\Rightarrow\text{STOP before A2/A3/B/C/F}.}
\]

## 11. 输出文件

根目录：`E0/outputs/e0_adp5/`

```text
adp5_0/final_metrics.json
a0_response_families/final_metrics.json
a0_response_families/data/visible_{train,val,test}.npz
a0_response_families/data/hidden_evaluator_only.npz
a1_family_discovery/final_metrics.json
a1_family_discovery/round_metrics.jsonl
a1_family_discovery/round_metrics.csv
a1_family_discovery/checkpoints/round_000...039_bank.pt
a1_family_discovery/assignments/round_000...039.npz
a1_family_discovery/final_latent_diagnostic.json
a1_family_discovery/a1_dynamics.png
a1_family_discovery/candidate_usage.png
```

代码与配置：

```text
E0/adp5/run_adp5.py
E0/adp5/diagnose_final_latent.py
E0/configs/e0_adp5_8gb.json
```

## 12. 科学结论

ADP5 支持：

\[
\boxed{\text{完整 response family 是消除 sign fragmentation 的有效证据单位。}}
\]

但它也揭示：

\[
\boxed{
\text{family-level candidate identity}
\neq
\text{跨 family 一致的 realization coordinate 与真实 mechanism function。}
}

如果继续下一版，最小问题不应再是添加 merge 或数量成本，而应检查如何让同一 response family 内以及跨 base 的 latent realization 坐标具有一致的顺序、方向和尺度。
