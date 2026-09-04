# E0-ADP10 实验报告：从 ADP6 开始的后半链路稳定性

## 1. 结论摘要

ADP10 已按执行方案完成 3-seed gate，并在通过后原样扩展到 10 seeds。

\[
\boxed{\text{9/10 seeds end-to-end PASS，达到预注册 strong pass（}\ge 8/10\text{）}}
\]

分阶段通过率：

| 阶段 | 通过率 |
|---|---:|
| ADP6-A1 coordinate / functional formation | 10/10 |
| ADP6-A2 functional stability | 10/10 |
| ADP9 automatic pruning + count recovery | 9/10 |
| B1 full-TP composition | 9/9 个进入 B1 的 seed |
| End-to-end | 9/10 |

因此，本实验支持：

\[
\boxed{
\text{固定可用的 ADP5 response-family evidence 后，ADP6→ADP9→B1 后半链路具有较高优化稳定性。}
}
\]

它仍不证明完整系统从最随机 candidate bank、且没有 family evidence 时也稳定。

唯一失败 seed 9 是 **F3 duplicate-graph false negative**：其第二组 duplicate 的三个跨域 \(R^2\) 仍超过 0.9998，但 direct-replacement \(\Delta\mathrm{NRMSE}=0.011423\)，略高于固定 gate 0.01，故算法按预注册规则停在 \(M=4\)。没有放宽阈值或事后补算 B1。

---

## 2. 实验配置与独立性

固定项：

```text
WORLD_SEED       = 20260901
DATASET_SEED     = 20260902
Mmax             = 5
mechanism dim    = 64
mechanism heads  = 4
ADP6 rounds      = 40
epochs / round   = 4
family batch     = 32
mechanism lr     = 3e-4
coordinate lr    = 1e-3
B1 inner steps   = 60
B1 restarts      = 3
B1 chunk size    = 1024
lambda_P         = 0.001
```

所有 seed 共用相同 E0 world、split、ADP5 response families、family assignment evidence 和 observable \(\alpha\)。与历史 development ADP6 不同的是，ADP10-A1 不复制 ADP5 mechanism checkpoint，而是在每个 optimization seed 下独立随机初始化 mechanism bank；这样才实际改变方案要求的 model initialization randomness。固定的是 family evidence，不是已学成的机制参数。

S0 审计结果：

| 审计项 | 结果 |
|---|---:|
| 不同 initial model hashes | 10/10 |
| 不同 optimization seeds | 10/10 |
| unique dataset hashes | 1 |
| unique family dataset hashes | 1 |

候选编号只在单 seed 内使用；跨 seed 的结论基于 functional identity、equivalence class 和 evaluator matching。

---

## 3. 每个 seed 的正式结果

表中的 F1 是 IID 与 101 participation micro-F1 的较小值；`—` 表示该 seed 被上游正式 gate 截停，未运行 B1。

| Seed | A1 | A2 functional | A2 min func \(R^2\) | A2 NRMSE | Pruning path | Final M | B1 IID | B1 101 | min F1 | Final |
|---:|---|---|---:|---:|---|---:|---:|---:|---:|---|
| 0 | PASS | PASS | 0.999427 | 0.02601 | 5→4→3 | 3 | 0.03051 | 0.03319 | 0.99682 | PASS |
| 1 | PASS | PASS | 0.999912 | 0.01028 | 5→4→3 | 3 | 0.02188 | 0.02204 | 0.99815 | PASS |
| 2 | PASS | PASS | 0.999884 | 0.01215 | 5→4→3 | 3 | 0.02444 | 0.02531 | 0.99763 | PASS |
| 3 | PASS | PASS | 0.999570 | 0.02028 | 5→4→3 | 3 | 0.02658 | 0.02967 | 0.99787 | PASS |
| 4 | PASS | PASS | 0.999905 | 0.00962 | 5→4→3 | 3 | 0.02283 | 0.02460 | 0.99774 | PASS |
| 5 | PASS | PASS | 0.999800 | 0.01453 | 5→4→3 | 3 | 0.02377 | 0.02380 | 0.99818 | PASS |
| 6 | PASS | PASS | 0.999451 | 0.02094 | 5→4→3 | 3 | 0.02639 | 0.03021 | 0.99820 | PASS |
| 7 | PASS | PASS | 0.999912 | 0.00999 | 5→4→3 | 3 | 0.02215 | 0.02270 | 0.99809 | PASS |
| 8 | PASS | PASS | 0.999452 | 0.02407 | 5→4→3 | 3 | 0.02874 | 0.03249 | 0.99814 | PASS |
| 9 | PASS | PASS | 0.999271 | 0.02544 | 5→4→stop | 4 | — | — | — | FAIL (F3) |

\[
RecoveryRate=\frac{9}{10}=0.90.
\]

最终机制数分布：

```text
M_discovered = 3 : 9 seeds
M_discovered = 4 : 1 seed
```

---

## 4. ADP6 稳定性

### A1

10/10 seeds 从独立随机机制参数出发通过 A1：

| 指标 | min | median | max |
|---|---:|---:|---:|
| held-out realization NRMSE | 0.01391 | 0.02019 | 0.02428 |

所有 seed 同时通过 order、affine-coordinate、held-out 和 purity-high functional gates。这说明在固定且正确的 response-family evidence 下，shared affine realization coordinate 和 mechanism function 并不依赖 seed-0 的历史 checkpoint。

### A2

| 指标 | min | median | max |
|---|---:|---:|---:|
| held-out realization NRMSE | 0.00962 | 0.01740 | 0.02601 |
| min matched functional \(R^2\) | 0.999271 | 0.999685 | 0.999912 |

A2 中 duplicate candidates 间仍存在明显 assignment 重排，某些 round 的 raw Hungarian accuracy/fragmentation 因一个 GT family 被两个等价 candidates 分割而较差。ADP10 按方案不把 duplicate symmetry 造成的振荡当作功能失败；10/10 seeds 的 active candidates 均保持高 purity，三个 GT functional identities 均被恢复，coordinate 和 structured held-out 均通过。

---

## 5. 自动构图与逐步剪枝

seeds 0–8 都自动形成两条正确 equivalence edges，并在每次 hard prune 后冻结 mechanism、全量 reassignment、重新验证，最终自然停在：

\[
5\rightarrow4\rightarrow3\rightarrow\text{no valid duplicate edge}.
\]

这 9 个 seed 的最终指标完全一致地表现为：

| 指标 | min | median | max |
|---|---:|---:|---:|
| family Hungarian accuracy | 1.0 | 1.0 | 1.0 |
| mean fragmentation | 0.0 | 0.0 | 0.0 |
| structured IID NRMSE | 0.00976 | 0.01455 | 0.02650 |
| min functional \(R^2\) | 0.999427 | 0.999800 | 0.999912 |

### seed 9 的失败定位

seed 9 第一条 C1+C4 edge 合法并成功删除一个 duplicate。另一组 C2+C5 的证据为：

| 指标 | 数值 | gate |
|---|---:|---:|
| cross-state \(R^2\) | 0.999884 | > 0.995 PASS |
| cross-realization \(R^2\) | 0.999846 | > 0.995 PASS |
| cross-base \(R^2\) | 0.999836 | > 0.995 PASS |
| direct-replacement max \(\Delta\)NRMSE | 0.011423 | < 0.01 **FAIL** |

其超阈值绝对量为 0.001423。因此这是一个边界性的 replacement gate false negative，而不是 functional identity 消失或错误 pair 被合并。正式结果仍严格记 FAIL。

---

## 6. Full-TP B1

9 个通过 pruning 的 seed 均使用各自 surviving mechanisms 独立完成完整的 20k IID 和 20k combination-101 exact-support inference。

| 指标 | min | median | max | gate |
|---|---:|---:|---:|---:|
| IID NRMSE | 0.02188 | 0.02444 | 0.03051 | < 0.05 |
| 101 NRMSE | 0.02204 | 0.02531 | 0.03319 | < 0.10 |
| IID participation micro-F1 | 0.99716 | 0.99809 | 0.99820 | > 0.90 |
| 101 participation micro-F1 | 0.99682 | 0.99844 | 0.99872 | > 0.90 |
| IID min active instance-effect \(R^2\) | 0.99826 | 0.99898 | 0.99920 | > 0.90 |
| 101 min active instance-effect \(R^2\) | 0.99863 | 0.99920 | 0.99941 | > 0.90 |
| 101 inactive-Z2 FPR | 0.00165 | 0.00280 | 0.00925 | < 0.10 |
| 101 \(E\|\hat e_2\|\) | 0.000061 | 0.000104 | 0.000321 | descriptive |

注意：101 中 GT \(Z_2\) 恒为 inactive，因此不对其计算无定义/无意义的 \(R^2\)；使用 FPR 和 predicted effect norm。

---

## 7. Negative controls

### NC-1：usage prune

10/10 seeds 的 lowest-usage two-step prune 都通过，包括正式 graph FAIL 的 seed 9。这个 control 在当前理想 family evidence 下没有区分力：duplicate assignment 的 usage 恰好足以暴露冗余。它不能替代 functional graph，因为这一性质尚未在更一般的上游条件下验证。

### NC-2：prune 后 joint mechanism fine-tuning

对每个 seed 的第一条 duplicate edge，使用 source-copy 初始化，并按 ADP7 的四个 cross-realization 方向训练/测试：

```text
weak -> strong
strong -> weak
negative -> positive
positive -> negative
```

40 个 split 中仅 21 个通过全部 ADP7-A2 条件；10/10 seeds 至少有一个方向失败。\(\Delta\)NRMSE 范围为 -0.02799 到 0.15657，中位数 0.00546。functional \(R^2\) 仍较高（最低 0.98823），说明本次跨 seed 复现的主要是 **cross-realization prediction drift**，而非每次都发生 functional identity 崩溃。该结果继续支持 ADP9 的“剪枝后冻结 mechanism”设计。

---

## 8. 计算代价与硬件

硬件：NVIDIA GeForce RTX 3070 Ti Laptop GPU，8GB；conda 环境 `wm`，PyTorch 2.5.1+cu121。

单独运行 seeds 0–3 时，每 seed 约 8–9 分钟；后续最多三个 seed 并行以提高 GPU 利用率，所以各进程内部 wall time 因资源共享增大，不宜相加解释为实际耗时。整个正式 10-seed 运行、controls 刷新和汇总从 15:47 到 17:30，约 102.5 分钟。

单进程代表性时间：

```text
A1: 约 88 s
A2: 约 90–93 s
B1: 约 305–358 s
```

A1/A2 单进程报告峰值约 47 MiB；三进程并行时通过 `nvidia-smi` 观察到总显存约 2.6GB、GPU 利用率约 93%。未因 8GB 显存限制缩小数据、round、epoch、B1 steps 或 restarts。

---

## 9. Reproducibility 与输出

实现：

```text
E0/adp10/run_adp10.py
E0/configs/e0_adp10_adp6.json
E0/configs/e0_adp10_adp9.json
```

输出：

```text
E0/outputs/e0_adp10_from_adp6/
├── seed_000/ ... seed_009/
│   ├── s0_audit/
│   ├── a1_fixed_assignment_structured_v/
│   ├── a2_joint_family_coordinate/
│   ├── adp9_0_equivalence_graph/
│   ├── prune_round_*/
│   ├── final_pruned_model/
│   ├── b1_full_tp/                 # 仅通过 pruning 的 seeds
│   ├── negative_controls/
│   └── final_verdict.json
├── aggregate_3seed/
└── aggregate_10seed/
```

复现实验：

```powershell
conda run -n wm python E0/adp10/run_adp10.py --stage seed_run --seed 0
conda run -n wm python E0/adp10/run_adp10.py --stage aggregate --seeds 0,1,2
conda run -n wm python E0/adp10/run_adp10.py --stage auto
```

完整性检查：

- [x] 10 个 optimization seeds 独立；initial hashes 全不同
- [x] world/data/family hashes 跨 seed 相同
- [x] ADP6/ADP9/B1 正式超参数跨 seed 相同
- [x] 第一阶段 3/3 后才扩展到 10 seeds
- [x] graph 不使用 GT 建边
- [x] 不以 alive count = 3 强制停止
- [x] pruning 后 mechanism 冻结
- [x] B1 统一使用 60 steps / 3 restarts
- [x] 失败 seed 未单独加预算或放宽阈值
- [x] seed 9 在上游 gate 后停止，未伪造 B1
- [x] JSON、checkpoint、assignment 和 round telemetry 已保存

---

## 10. 最终科学判定

\[
\boxed{
\textbf{ADP10 STRONG PASS：后半链路恢复率 9/10。}
}
\]

更精确的表述是：

> 在 ADP5 已提供高质量 response-family identity evidence 的条件下，从随机 ADP6 mechanism initialization 开始，shared realization coordinate、functional duplicate discovery、冻结机制的逐步剪枝以及 full-TP composition 在 10 个 optimization seeds 中有 9 个完整成功。当前剩余不稳定点主要是 duplicate graph 的 hard replacement threshold，而不是 coordinate formation、functional identity 或 full-TP composition。

下一步适合按方案进入 from-zero stability under perfect TP，并保留 identity TP vs orthogonal TP 对照；若要先修复后半链路，优先研究 replacement gate 的不确定性/置信区间，而不是事后直接把 0.01 调宽。
