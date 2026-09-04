# E0-ADP6 实验报告：统一 realization 坐标的形成

## 1. 最终结论

ADP6 的核心 A1 实验 **PASS**。在固定 ADP5 family assignment、只把逐点自由 latent 改成 candidate-shared affine 坐标

\[
v_{U,r}=a_{c_U}\tilde\alpha_{U,r}+b_{c_U}
\]

后，held-out-realization NRMSE 从 ADP5 的 `0.183116` 降到 `0.018561`，三个 matched functional \(R^2\) 从全负恢复到

\[
[0.999810, 0.999873, 0.999705].
\]

这说明 ADP5 的主要失败确实来自 realization gauge 未统一；observable action coordinate 足以让固定 family identity 下的真实 functional mechanism 成形。

但恢复 assignment learning 后，A2 **FAIL**：函数身份和预测仍近乎完美，重复 candidate 之间的 family assignment 却持续交换。最终 family accuracy 为 `0.784974`，mean fragmentation 为 `0.217700`，last-3 assignment change 为 `0.232303`，均未通过 A2 阈值。按执行方案的 stage gate，本次停止于 A2，没有执行 A3、B、C。

因此当前最准确的结论是：

\[
\boxed{\text{realization coordinate 问题已解决；duplicate identity / assignment stability 尚未解决。}}
\]

## 2. 环境、数据与控制变量

- 环境：Conda `wm`，Python 3.10.19，PyTorch 2.5.1+cu121，CUDA 可用。
- GPU：NVIDIA GeForce RTX 3070 Ti Laptop GPU，8 GB。
- world seed：`20260901`；dataset seed：`20260902`；optimization seed：`0`。
- 数据沿用 ADP5：train/val/test bases = `1024/256/512`，\(K_g\in\{1,2\}\)，\(M_{\max}=5\)。
- train realization：`[-1.0,-0.7,-0.4,+0.4,+0.7,+1.0]`。
- held-out realization：`[-0.85,-0.55,-0.3,+0.3,+0.55,+0.85]`。
- mechanism architecture：5 candidates，dimension 64，4 heads。
- 优化：AdamW；mechanism LR `3e-4`；coordinate LR `1e-3`；weight decay `1e-5`；batch 32 families；每轮 4 epochs；40 rounds；gradient clip 1.0。
- A1 初始化：ADP5 A1 最终失败 checkpoint；family assignment 固定为 ADP5 最终 assignment。
- A1 唯一研究性改动：取消 per-point free \(v\)，使用 candidate-level shared affine map；不使用 GT amplitude 监督。
- 8 GB 调整：没有缩小科学规模。A1 显存峰值仅 `48.36 MiB`，A2 为 `48.59 MiB`。

PC1 的符号本身不唯一。本实现采用 learner-visible 确定性约定：令 family 中第一个非零 action member 在 \(q_U\) 上的投影为正。没有用 GT amplitude 决定方向。

## 3. ADP6-0：action-coordinate 审计

| 指标 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| median \(|\mathrm{Spearman}|\) | 1.000 | > 0.98 | PASS |
| median \(|\mathrm{Kendall}|\) | 1.000 | > 0.95 | PASS |
| median pairwise order accuracy | 1.000 | > 0.98 | PASS |
| global sign agreement after one flip | 1.000 | 诊断项 | — |

observable action geometry 在本 synthetic world 中精确给出了 realization 的相对顺序和尺度，因此允许进入 A1。

## 4. A1 最终结果与判据

| 指标 | ADP5 free-v | ADP6 shared affine | A1 阈值 | 判定 |
|---|---:|---:|---:|---|
| held-out NRMSE | 0.183116 | **0.018561** | < 0.15 | PASS |
| matched functional \(R^2\) | -0.481/-0.488/-0.642 | **1.000/1.000/1.000** | 至少两个 purity-high > 0.70，且不能全负 | PASS |
| median order accuracy | — | **1.000** | > 0.95 | PASS |
| min candidate affine-aligned \(R^2\) | — | **1.000** | > 0.90 | PASS |
| relative functional improvement | baseline | 极大提高 | 必须明显提高 | PASS |

最终 candidate 参数：

```text
a = [0.954657, 0.979190, 0.984228, 0.958264, 0.981467]
b = [-0.045634, 0.049415, 0.031737, -0.045084, 0.052409]
```

五个 purity-high candidates 的 functional \(R^2\) 均在 `0.9996` 以上。它们仍保持 ADP5 的重复结构：C1/C4 对应同一个 GT，C2/C5 对应另一个 GT，C3 单独对应第三个 GT。A1 的目标不是 merge，因此 candidate 数仍为 5 不构成失败。

A1 40 轮耗时 `93.63 s`，每轮末端约 `1.87 s`。NRMSE 从 round 0 的 `0.09337` 下降至 round 39 的 `0.01856`，全程最小值 `0.01579`。

## 5. Negative controls

| 方法 | assignment | realization 参数化 | held-out NRMSE | matched functional \(R^2\) |
|---|---|---|---:|---|
| ADP5 free-v | fixed/learned | per-point free | 0.183116 | -0.481/-0.488/-0.642 |
| ADP6 A1 | fixed | candidate-shared affine | **0.018561** | **1.000/1.000/1.000** |
| NC-1 shuffle alpha | fixed | family 内随机置乱 | 1.253033 | -0.481/-0.363/-0.433 |
| NC-2 family flip | fixed | family 独立随机反向 | 1.207645 | -0.117/-0.091/-0.089 |
| NC-3 family affine | fixed | family-specific affine | 0.021848 | 1.000/1.000/1.000 |

NC-1 和 NC-2 明确失败，排除了“仅靠额外训练轮数即可恢复”的解释，并证明正确顺序与跨-family方向一致性是有效信号。

NC-3 没有失败。其 validation family affine 参数是在冻结 bank 后，仅用 observed realization responses 做 40 步 learner-visible 内循环拟合；held-out realization 没有参与拟合。最终 validation \(a_U\) 的标准差仅 `0.00950`，\(b_U\) 的标准差仅 `0.00791`。这表明本 synthetic world 的 action coordinate 已经天然全局对齐，family-specific 参数从 \(a=1,b=0\) 初始化后没有产生明显 gauge 漂移。因此本实验支持“正确统一坐标很重要”，但**不能单独证明 candidate-shared affine 是唯一或必要的参数化**。

## 6. A2：恢复 assignment learning

A2 每轮执行：冻结 bank/coordinate，按完整 family response 枚举 5 个 candidate 的 structured-coordinate cost 并取 argmin；随后冻结 assignment，更新 mechanism 与 \(a_j,b_j\)。没有 per-point latent inner optimization，merge 关闭，global count cost 为 0。

最终结果：

| 指标 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| family Hungarian accuracy | 0.784974 | > 0.85 | FAIL |
| mean fragmentation | 0.217700 | < 0.20 | FAIL |
| last-3 assignment change | 0.232303 | < 0.15 | FAIL |
| min matched functional \(R^2\) | 0.999883 | > 0.75 | PASS |
| matched functional \(R^2>0.90\) count | 3 | >= 2 | PASS |
| min affine-aligned realization \(R^2\) | 1.000 | > 0.90 | PASS |
| median order accuracy | 1.000 | > 0.95 | PASS |
| held-out NRMSE | 0.011963 | 诊断项 | — |

40 轮中 assignment change 平均为 `0.272436`，family accuracy 在 `0.709845–0.958549` 之间来回波动。最终 contingency 为：

```text
          GT1  GT2  GT3
C1          0    0   93
C2         77    0    0
C3          0  133    0
C4          0    0   31
C5         52    0    0
```

每个 candidate 的 purity 都是 1.0；失败不是 mechanism mixing，而是 GT1 与 GT3 分别在两个功能几乎完全相同的 duplicate candidates 之间重分配。由于两个副本的 cost 几乎相同，微小优化噪声即可改变 argmin，造成 assignment 大幅振荡。该现象对应执行方案的“情况 3”：coordinate 已解决，但 identity assignment dynamics 仍不稳定。

## 7. Stage-gate 决策

- ADP6-0：PASS，进入 A1。
- A1：PASS，完成 negative controls 并进入 A2。
- A2：FAIL，`stop_required=true`。
- A3：未执行；原因是 A2 gate 未通过。
- B、C：未执行；它们分别依赖 A3、B 通过。

不能把未执行阶段记为 FAIL；它们是按预注册 gate 合法停止。

## 8. Windows DLL 异常诊断

此前 `spearmanr -> numpy.corrcoef`、`numpy.linalg.svd` 出现 Windows `0xc06d007f`。复查发现 MKL DLL 文件存在；根因是直接调用 `E:\conda\envs\wm\python.exe` 时，进程 PATH 未包含 `E:\conda\envs\wm\Library\bin`。显式加入该目录或使用 `conda run -n wm` 后，`corrcoef` 与 BLAS 路径均正常。

因此没有下载 DLL，也没有重装/更改 NumPy、SciPy、MKL 或 PyTorch。这样避免了在实验中途改变数值栈。`run_adp6.py` 入口已加入 Conda DLL 目录初始化，并提供不受 PowerShell execution policy 限制的 `run_adp6_wm.cmd` 作为标准启动器。

推荐命令：

```bat
"E:\codes_python\relation latent world\E0\adp6\run_adp6_wm.cmd" adp6_a2 0
```

或：

```powershell
E:\conda\Scripts\conda.exe run -n wm --no-capture-output python -X faulthandler E0\adp6\run_adp6.py --stage adp6_a2 --seed 0
```

## 9. 输出与复现信息

```text
E0/outputs/e0_adp6/
├── adp6_0_action_coordinate/
├── a0_free_v_baseline/
├── a1_fixed_assignment_structured_v/
│   ├── checkpoints/round_000.pt ... round_039.pt
│   ├── round_metrics.jsonl
│   └── final_metrics.json
├── negatives/
│   ├── shuffle_alpha/
│   ├── family_flip/
│   └── family_affine/
└── a2_joint_family_coordinate/
    ├── checkpoints/round_000.pt ... round_039.pt
    ├── assignments/round_000.npz ... round_039.npz
    ├── round_metrics.jsonl
    └── final_metrics.json
```

关键复现项：

- [x] 固定 world/dataset/optimization seeds。
- [x] 保存完整配置、环境版本、Git commit、输入 checkpoint 路径与哈希。
- [x] A1/A2 每轮保存 telemetry 与 checkpoint。
- [x] A2 每轮保存 train/val assignment 及完整 candidate cost。
- [x] 保存 wall-clock 与 CUDA peak allocation。
- [x] negative controls 使用确定性随机种子。
- [x] A1 实际运行代码快照 SHA256 为 `35709c59...068b90`；运行后扩展 NC/A2 时保留了该快照。
- [x] 遵守 stage gate，没有在 A2 FAIL 后继续 A3/B/C。

## 10. 下一步研究含义

不应再修改 realization coordinate，也不应回到 per-point free \(v\)。下一步应只针对 duplicate candidates 的 assignment symmetry：在不破坏已恢复的 functional identity 与 coordinate gauge 的前提下，引入明确的 duplicate merge 或 assignment tie stabilization。当前证据显示问题已从“机制学不出来”收缩为“两个等价机制副本之间没有稳定的全局身份选择”。
