# E0 实验最终报告

**最终判定：E0 FAIL**

E0-A did not reach >=8/10 full structural recovery for both TP coordinate systems; E0-B is gated.

## 1. Experiment validity

- Oracle maximum MSE: 0.000e+00（阈值 < 1e-10，PASS）
- held-out 101 泄漏：False
- Identity TP condition number: 1（identity）
- Orthogonal TP condition number: 1.000000
- Ill-conditioned control condition number: 10000.18
- 数据规模：train 100,000；val/test-IID/test-context/test-101 各 20,000。
- 每个正式 seed 运行 200 个记录周期 × 64 个随机 minibatches = 12,800 optimizer updates。
- learner training path 只读取 visible.npz；hidden_gt.npz 只由 oracle/evaluation 打开。

## 2–3. E0-A identity / orthogonal

| TP | full PASS seeds | IID NRMSE | 101 NRMSE | participation F1 | min functional R² | redundant usage |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0/10 | 0.1002 ± 0.0068 | 0.1254 ± 0.0116 | 0.6850 | -1.2663 | 1.0000 |
| orthogonal | 0/10 | 0.1036 ± 0.0082 | 0.1289 ± 0.0146 | 0.6774 | -1.4234 | 1.0000 |

预注册阈值：IID NRMSE < 0.05，101 NRMSE < 0.1（且 predictive pass 要求 < 0.05），F1 > 0.9，functional R² > 0.9，redundant usage < 0.1。

两种 TP 均出现一致退化：几乎所有 candidate gate 全开，重构持续改善，但没有恢复 GT participation 或 mechanism function。

## 4. E0-B relation

未运行。执行指南规定只有 E0-A PASS 后才能生成/训练 E0-B；当前不存在可解释的 recovered Z_D 基座。
因此 relation EP、方向性和 relation functional R² 均为 N/A，而不是 0。

## 5. Negative controls

| Control | IID NRMSE mean | 101 NRMSE mean | 结果 |
|---|---:|---:|---|
| NC1_Mmax_lt_Mstar | 0.4116 | 0.5127 | 0/3 PASS |
| NC2_factorial_removed | 0.1485 | 0.1969 | 0/3 PASS |
| NC4_shuffle_p | 1.0029 | 1.0030 | 0/3 PASS |
| NC5_lambda_P_zero | 0.1438 | 0.1718 | 0/3 PASS |
| NC7_ill_conditioned_tp | 0.5206 | 0.6595 | 0/3 PASS |
| NC3_no_relation_learner | N/A | N/A | E0-B is gated by E0-A failure; no recovered Z_D base exists |
| NC6_directional_variation_removed | N/A | N/A | directional edge experiment is gated by E0-A failure |

## 6. Pass / Fail

| Gate | Result | Reason |
|---|---|---|
| Pass 0 exact realizability | PASS | oracle MSE = 0 |
| Pass 1 predictive recovery | FAIL | identity/orthogonal IID NRMSE 均值均 > 0.05 |
| Pass 2 structural recovery | FAIL | F1、functional R²、redundancy 均未达标 |
| Pass 3 compositional recovery | FAIL | 101 NRMSE 均值均 > 0.1 |
| Pass 4 relation recovery | N/A / prerequisite FAIL | E0-A 未通过，relation stage 被门控 |
| Pass 5 optimization stability | FAIL | identity 0/10，orthogonal 0/10 |

## 结论

在 generator 严格可实现且数据 coverage 充分时，当前 objective、parameterization 与 optimization implementation 没有稳定恢复正确的 reusable mechanism decomposition。
主要失败不是 TP 坐标依赖，而是 participation complexity 没有阻止冗余全开解；预测拟合也未达到阈值。
该结论仅适用于本指南定义的 synthetic E0，不外推到真实世界。