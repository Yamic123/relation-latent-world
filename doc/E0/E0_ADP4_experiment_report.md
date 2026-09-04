# E0-ADP4 开发阶段实验报告

## 1. 结论

实验严格按 `Adp4_experiment_execution_plan.md` 的 stage gate 执行：

| Stage | 结果 | 是否继续 |
|---|---|---|
| Adp4-0：复核 local-unit 结构 | PASS | 进入 A0 |
| Adp4-A0：observable fragments 自动形成 local units | PASS | 进入 A1 |
| Adp4-A1：base 内 injective joint matching | **FAIL** | **STOP** |
| A2/A3/B/F | 未执行 | A1 gate 禁止继续 |

核心结论：

\[
\boxed{
\text{local unit + base 内 injective matching 显著稳定了 assignment，}
\text{但把跨 base 的 }v\text{-sign fragmentation 固化得更彻底。}
}
\]

A1 最终 mean sign-split 为 1.000，表示同一 GT mechanism 的正、负 realization 几乎被完全分配给不相交的 candidate 集合。三个 matched functional \(R^2\) 仅为：

\[
[0.3595,\ 0.1608,\ 0.1704].
\]

因此不能进入旨在“合并已成形 family 副本”的 A2：当前 candidate 首先是 sign-specific fragments，而不是已经正确形成、仅有冗余的完整 mechanism family。

## 2. 环境与控制变量

训练与数值评价使用：

```text
conda env: wm
Python 3.10.19
PyTorch 2.5.1+cu121
GPU: NVIDIA RTX 3070 Ti Laptop, 8 GB
optimization seed: 0
```

保持原 E0/ADP3 条件：

```text
WORLD_SEED = 20260901
DATASET_SEED = 20260902
M_real = 3（evaluator only）
M_max = 5
SetMechanismBank dim = 64, heads = 4
TP = identity
R_D = 0
```

正式实验没有缩减科学规模。唯一显存控制是将 differentiable inner solve 按 256 local units 分块。

```text
inner Adam lr = 0.05
inner steps = 20
restarts = 2
v = 1.5 tanh(z)
M-step AdamW lr = 3e-4
weight decay = 1e-5
batch = 128 local units
epochs/round = 4
rounds = 40
```

## 3. Adp4-0：只读结构复核

| 指标 | 结果 |
|---|---:|
| 所有 base 是否均为 3×4 | 是 |
| Within-family max distance | \(1.3411\times10^{-7}\) |
| Between-family min distance | 0.154881 |
| Separation ratio | \(8.659\times10^{-7}\) |

同一 base、同一 GT mechanism 的四条 contrast 在数值精度内重复，且与其它 family 清楚分离。Adp4-0 PASS。

## 4. Adp4-A0：无需 GT 的 local-unit 构造

阈值完全由 learner-visible train fragments 得到：

\[
\tau_{unit}=\max(10^{-6},10Q_{0.999}(D_{nearest}))
=1.17923\times10^{-6}.
\]

| Split | Bases | Units | 恰好 3 units/base | Purity | Completeness |
|---|---:|---:|---:|---:|---:|
| Train | 1024 | 3072 | 1.000 | 1.000 | 1.000 |
| Val | 256 | 768 | 1.000 | 1.000 | 1.000 |
| Test | 512 | 1536 | 1.000 | 1.000 | 1.000 |

Adp4-A0 PASS。这证明 local unit 可以只从 `group_id, S, d` 自动恢复，无需 GT mechanism label。

## 5. Adp4-A1 最终指标

| 判据 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| Unit Hungarian accuracy | 0.5404 | >0.80 | FAIL |
| 三个 matched functional \(R^2\) | [0.3595, 0.1608, 0.1704] | 每个 >0.70 | FAIL |
| \(R^2>0.85\) 的数量 | 0 | 至少 2 | FAIL |
| Last-3 base assignment change | 0.1068 | <0.15 | PASS |
| Mean sign-split | 1.0000 | <0.25 | FAIL |

其它指标：

```text
final val local-unit NRMSE = 0.10447
final base exact restart agreement = 0.86719
formal A1 wall time = 301.08 s
PyTorch peak allocated GPU memory = 48.93 MiB
```

预测误差和 assignment 稳定性很好，但 mechanism identity 明确失败。这正是方案中定义的“假成功”。

## 6. 最终 candidate 分配

Train usage：

| Candidate | Usage | Unit count |
|---|---:|---:|
| C1 | 0.1732 | 532 |
| C2 | 0.1813 | 557 |
| C3 | 0.2507 | 770 |
| C4 | 0.2214 | 680 |
| C5 | 0.1735 | 533 |

Val contingency（row=candidate，column=GT family）：

| Candidate | Z1 | Z2 | Z3 |
|---|---:|---:|---:|
| C1 | 0 | 134 | 0 |
| C2 | 0 | 120 | 18 |
| C3 | 117 | 0 | 59 |
| C4 | 134 | 0 | 32 |
| C5 | 5 | 2 | 147 |

表面上每个 candidate 具有一定 family purity，但每个真实 family 被多个 candidate 分担；其分界主要由 realization sign 决定。

## 7. (v)-sign split 的直接证据

最终条件分布：

```text
Z1, v<0: C4 = 100%
Z1, v>0: C3 = 95.9%, C5 = 4.1%

Z2, v<0: C1 = 100%
Z2, v>0: C2 = 98.4%, C5 = 1.6%

Z3, v<0: C5 = 100%
Z3, v>0: C2 = 16.5%, C3 = 54.1%, C4 = 29.4%
```

三个 GT family 的 sign-split 均为 1.0。ADP3-A1 最终 mean sign-split 约为 0.770；ADP4-A1 不但没有减弱它，反而把它提高到了完全分裂。

组内 injective matching 只保证：

\[
\text{同一 base 的三个不同 local effects 使用三个不同 candidate}.
\]

但每个 base 对每个 GT mechanism 只有一个固定 realization sign。该约束没有提供任何跨 base 的证据，要求正、负 realization 必须属于同一个 global candidate。随着 assignment 变稳定，这种 sign-specific permutation 也一起被稳定下来。

## 8. Functional identity

最终 functional \(R^2\) matrix：

| Candidate | GT Z1 | GT Z2 | GT Z3 |
|---|---:|---:|---:|
| C1 | -0.4461 | -0.2549 | -0.3798 |
| C2 | -0.1872 | **0.1608** | 0.2200 |
| C3 | **0.3595** | -0.0913 | 0.4358 |
| C4 | 0.0468 | -0.1653 | 0.0961 |
| C5 | -0.0921 | -0.0832 | **0.1704** |

虽然某些非 Hungarian 单元格略高，例如 C3→Z3 为 0.4358，但一对一匹配后没有任何 GT family 达到 0.70。没有形成可供 A2 安全合并的完整 function family。

## 9. 与 Adp3-A1 的直接对照

| 方法 | Val NRMSE | Hungarian accuracy | Matched functional \(R^2\) | Last-3 change | Mean sign-split |
|---|---:|---:|---|---:|---:|
| Adp3-A1 | 0.1459 | 0.5674 | [0.8886, 0.3232, -0.1760] | 0.3197 | 0.7699 |
| Adp4-A1 | **0.1045** | 0.5404 | [0.3595, 0.1608, 0.1704] | **0.1068** | **1.0000** |

ADP4 改善了 reconstruction 与稳定性，但没有改善 structure recovery，且 sign fragmentation 更严重。

## 10. 为什么停止而不进入 A2

A2 的 pairwise union test 假设输入是同一真实 mechanism 的 realization 碎片，且至少已有可识别的完整 family。当前 A1 的三个主要 identity 指标均失败，直接进入 A2 会把“修复身份形成”与“模型数量选择”混在一起。

因此严格执行：

\[
\boxed{\text{A1 FAIL}\Rightarrow\text{STOP before A2}.}
\]

## 11. 输出

根目录：`E0/outputs/e0_adp4/`

```text
adp4_0/final_metrics.json
a0_local_units/final_metrics.json
a0_local_units/data/visible_{train,val,test}.npz
a0_local_units/data/hidden_gt.npz
a1_group_matching/final_metrics.json
a1_group_matching/round_metrics.jsonl
a1_group_matching/round_metrics.csv
a1_group_matching/checkpoints/round_000...039_bank.pt
a1_group_matching/assignments/round_000...039.npz
a1_group_matching/a1_dynamics.png
a1_group_matching/candidate_usage.png
```

复现脚本：`E0/adp4/run_adp4.py`；配置：`E0/configs/e0_adp4_8gb.json`。
