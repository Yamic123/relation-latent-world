# E0-ADP9 实验报告：基于功能等价的逐步删冗余机制

## 1. 最终结论

ADP9 在 development seed 0 上完整 **PASS**：

\[
\boxed{
5\ \text{candidates}
\rightarrow 4
\rightarrow 3
\rightarrow \text{no valid duplicate edge, STOP}
}
\]

整个 pruning 过程没有读取 \(M_{\rm real}=3\)，没有硬编码 duplicate pairs，也没有更新任何 mechanism 参数。learner 从全部 10 个 pair 自动构图，逐轮选择：

```text
round 0: C2+C5，保留 C2，删除 C5
round 1: C1+C4，保留 C1，删除 C4
stop:    C1/C2/C3 之间不存在 valid duplicate edge
```

最终：

- \(M_{\rm discovered}=3\)；
- family Hungarian accuracy = `1.0`；
- mean fragmentation = `0.0`；
- 三个 functional \(R^2\) 均大于 `0.99988`；
- full-TP B1 在完整 20k IID 和 20k 101 上也通过。

因此 seed 0 支持：

\[
\boxed{
\text{response family}
\rightarrow
\text{统一 realization coordinate}
\rightarrow
\text{functional duplicate pruning}
\rightarrow
Z_D
}
\]

需要限定：目前只有独立的上游 ADP6-A2 seed-0 checkpoint，所以 multi-seed 和 8/10 strong pass 尚未评估。

## 2. 实验设置与约束

- 输入 checkpoint：ADP6-A2 `round_039.pt`。
- 输入 assignments：ADP6-A2 `round_039.npz`。
- world seed：`20260901`；dataset seed：`20260902`；development seed：`0`。
- 初始 alive：`C1,C2,C3,C4,C5`。
- learner 不读取 GT pair、GT family 或 \(M_{\rm real}\)。
- GT 只用于 prune 后 evaluator functional/assignment 指标。
- 所有 mechanism 参数全程冻结；state hash 在独立 verify 中保持一致。
- 删除时不做 averaging、interpolation、新 merged network 或 mechanism fine-tuning。
- 本轮 direct prune 已满足 gate，所以没有触发可选的 coordinate-only 校准。

## 3. ADP9-0：自动功能等价图

duplicate edge 要同时满足：

```text
cross-state prediction R² > 0.995
cross-realization prediction R² > 0.995
cross-base prediction R² > 0.995
direct-replacement max NRMSE increase < 0.01
```

全部 10 个 pair 的结果：

| Pair | Cross-S \(R^2\) | Cross-v \(R^2\) | Cross-base \(R^2\) | max replacement ΔNRMSE | Edge |
|---|---:|---:|---:|---:|---|
| C2+C5 | 0.999987 | 0.999988 | 0.999987 | 0.000283 | **是** |
| C1+C4 | 0.999967 | 0.999970 | 0.999969 | 0.000435 | **是** |
| C1+C2 | -2.0524 | -1.9525 | -2.2417 | 1.2011 | 否 |
| C1+C3 | -3.7483 | -2.7795 | -2.8337 | 1.1969 | 否 |
| C1+C5 | -2.0814 | -2.0825 | -2.2457 | 1.2297 | 否 |
| C2+C3 | -2.8107 | -2.3809 | -2.5546 | 1.2968 | 否 |
| C3+C5 | -2.0900 | -2.0269 | -2.2133 | 1.3170 | 否 |
| C4+C5 | -2.1920 | -2.1941 | -2.2577 | 1.3594 | 否 |
| C2+C4 | -2.1184 | -2.1173 | -2.2343 | 1.3749 | 否 |
| C3+C4 | -2.3243 | -2.3024 | -2.4377 | 1.4587 | 否 |

构图具有很高区分度，只产生两条有效 edge，而且其排序由 learner-visible equivalence confidence 决定。

## 4. Round 0：删除 C5

最高置信 edge 为 `C2+C5`。

survivor quality：

| Candidate | heldout loss | instability | coverage penalty | \(Q\) |
|---|---:|---:|---:|---:|
| C2 | 0.00000581 | 0.15285 | 0.15086 | **0.07583** |
| C5 | 0.00000629 | 0.15285 | 0.19324 | 0.08855 |

因此保留 C2、删除 C5。第一次 reassignment 相对旧 assignment 改变 `0.13472`，之后两次重复 reassignment change 都是 `0.0`。

单步 gate：

- max structured NRMSE increase：`0.0000519 < 0.01`；
- 三个 best functional \(R^2>0.95\)；
- family accuracy 下降 `<0.03`；
- fragmentation 增量 `<0.03`；
- repeated assignment change `0.0 < 0.10`。

全部通过，commit deletion。

## 5. Round 1：删除 C4

重新 assignment、重新构图后，最高置信 edge 为 `C1+C4`。

| Candidate | heldout loss | instability | coverage penalty | \(Q\) |
|---|---:|---:|---:|---:|
| C1 | 0.00001044 | 0.08549 | 0.12709 | **0.05523** |
| C4 | 0.00001023 | 0.08549 | 0.24844 | 0.09164 |

虽然 C4 的局部 heldout loss略低，但其 coverage penalty 更大，所以综合质量选择 C1。第一次 reassignment change 为 `0.08031`，随后两次为 `0.0`。

max structured NRMSE increase 仅 `0.0001190`，其余 gate 全部通过，commit deletion。

## 6. 自动停止与最终 family 指标

第二次 commit 后重新构图，`C1,C2,C3` 之间没有任何 valid duplicate edge，因此停止。代码中不存在 `alive_count == 3` 停止规则；数字 3 只在结束后由 evaluator 检查。

最终 family contingency：

```text
          GT1  GT2  GT3
C1          0    0  124
C2        129    0    0
C3          0  133    0
```

| 指标 | 结果 | 阈值 | PASS |
|---|---:|---:|---|
| IID heldout NRMSE | 0.012071 | < 0.03 | 是 |
| Cross-state NRMSE | 0.012587 | < 0.05 | 是 |
| Cross-realization NRMSE | 0.013955 | < 0.05 | 是 |
| Cross-base NRMSE | 0.012195 | < 0.05 | 是 |
| Functional \(R^2\) | 0.999930 / 0.999948 / 0.999883 | 全部 > 0.95 | 是 |
| Family accuracy | 1.000 | > 0.90 | 是 |
| Mean fragmentation | 0.000 | < 0.10 | 是 |
| Repeated assignment change | 0.000 | < 0.10 | 是 |

最终 candidate usage 为 C1 `0.3212`、C2 `0.3342`、C3 `0.3446`。

## 7. Negative controls

| Method | 结果 | 解释 |
|---|---|---|
| ADP9 gradual prune | PASS，最终 M=3 | learner-visible graph 正确识别并安全删除两项 |
| Random delete | 20 次中 9 次成功，pass rate `45%` | 仅靠随机删到 3 个不可靠 |
| Usage delete | PASS；删除 C4、C5 | 当前 seed 中最低 usage 恰好是 duplicates，因此此对照没有区分力 |
| Wrong-pair forced delete | FAIL | 强删唯一 C3 后 IID NRMSE `0.8475`、accuracy `0.5648` |
| Joint fine-tuning | FAIL | 复用 ADP8，cross-realization drift |
| Batch delete | PASS；同时删除 C5、C4 | 本 E0 seed 未证明逐步删除优于 batch delete |

### 对 controls 的准确解释

本实验支持 functional equivalence gate，不能声称“usage 删除或 batch 删除必然失败”。在这个非常干净的 seed 上，它们碰巧得到相同最终集合。逐步策略的优势仍是每次具备 rollback 能力和可解释 telemetry，而不是本轮表现出更低最终误差。

## 8. B1：回到完整 TP participation

pruning PASS 后，使用 alive=`C1,C2,C3` 枚举 \(2^3=8\) 个 support，并优化每个 active mechanism 的 \(\alpha\)：

\[
v_j=a_j\alpha_j+b_j.
\]

使用完整 `test_iid=20,000` 和 `test_combination_101=20,000`，没有缩小数据。

| 指标 | IID | 101 | 阈值 |
|---|---:|---:|---:|
| NRMSE | **0.024040** | **0.025966** | <0.05 / <0.10 |
| Participation micro-F1 | **0.997316** | **0.997489** | >0.90 |
| Active-GT macro-F1 | 0.997175 | 0.999036 | 诊断项 |
| Support exact accuracy | 0.99155 | 0.99050 | 诊断项 |
| Min active instance-effect \(R^2\) | 0.998993 | 0.999143 | >0.90 |
| Min functional \(R^2\) | 0.999883 | 0.999883 | >0.90 |
| Inactive Z2 FPR | — | **0.00620** | <0.10 |
| Inactive Z2 predicted effect norm | — | 0.000233 | 诊断项 |

B1 全部正式 checks 通过。

### Participation F1 定义

101 中 Z2 恒为 inactive，不能对这个全负 GT 类计算普通 positive-class F1；只要出现一个 false positive，该类 F1 就会变成 0。主 participation 指标因此采用所有 sample×mechanism bits 的 micro-F1，同时报告 active-GT macro-F1，并用预注册的 inactive Z2 FPR 单独衡量全负类。这避免了把 `FPR=0.0062` 错写成“F1=0 导致整体失败”。

## 9. B1 solver convergence

最初 30 steps / 2 restarts pilot 得到 IID/101 NRMSE `0.09166/0.09258`。但其 participation 与 effect \(R^2\) 已较高，提示连续求解未收敛。

在固定前 2048 样本上做 solver sanity：60 steps / 3 restarts 将 IID/101 NRMSE 降至 `0.02449/0.02499`。由于 ADP9 方案没有预注册 B1 solver steps/restarts，正式运行在查看 full-run最终判定前依据该收敛检查固定为 60/3，并重新运行完整 20k+20k。

pilot 和 convergence sanity 均独立保留，避免隐藏失败配置。

正式 B1 用时 `712.29 s`，CUDA peak allocation `96.87 MiB`；8 GB GPU 余量充分。

## 10. Multi-seed 状态

方案要求 seed 0 PASS 后测试 seeds 0/1/2。但 ADP9 的输入是上游 ADP6-A2 checkpoint 和 assignments，目前只存在 seed 0。

不能把同一 seed-0 checkpoint 换一个 ADP9 RNG 后称为独立 recovery seed。因此：

```text
seed-0 development: PASS
2/3 multi-seed:     NOT ASSESSED
8/10 strong pass:   NOT ASSESSED
```

完成 multi-seed 需要先生成 ADP6-A2 seeds 1、2；正式 10-seed 则需要完整的十组独立上游 checkpoints。

## 11. 输出结构

```text
E0/outputs/e0_adp9/
├── adp9_0_equivalence_graph/
│   ├── graph.json
│   └── final_metrics.json
├── prune_round_00/
│   ├── graph.json
│   ├── prune_decision.json
│   ├── pre_metrics.json
│   ├── post_metrics.json
│   └── assignments.npz
├── prune_round_01/
├── final_pruned_model/
│   ├── model.pt
│   ├── assignments.npz
│   ├── prune_metrics.json
│   └── final_verdict.json
├── negatives/
├── b1_full_tp/
│   ├── final_metrics.json
│   ├── test_iid_assignments.npz
│   ├── test_combination_101_assignments.npz
│   ├── solver_pilot_30x2.json
│   └── solver_convergence_sanity.json
├── multiseed_status.json
└── final_verdict.json
```

## 12. Reproducibility checklist

- [x] learner graph 没有硬编码 GT pair。
- [x] 停止条件是无 valid duplicate edge，不是 alive count。
- [x] 所有 mechanism 参数冻结并验证 state hash。
- [x] 每轮只删一个 candidate，并保存 pre/post/decision/assignments。
- [x] 每轮删除后重新 assignment、重新构图。
- [x] survivor 按预定 learner-visible \(Q_j\) 选择。
- [x] 所有单步 gate 通过后才 commit；代码支持 rollback/blacklist。
- [x] 保存全部 pair graph，而不只保存成功 edges。
- [x] 完成五组 seed-0 negative controls。
- [x] B1 使用完整测试集并保存 assignments。
- [x] 未把缺失的 upstream seeds 冒充 multi-seed 结果。

命令：

```bat
"E:\codes_python\relation latent world\E0\adp9\run_adp9_wm.cmd" adp9_0
"E:\codes_python\relation latent world\E0\adp9\run_adp9_wm.cmd" adp9_prune
"E:\codes_python\relation latent world\E0\adp9\run_adp9_wm.cmd" adp9_verify
"E:\codes_python\relation latent world\E0\adp9\run_adp9_wm.cmd" adp9_negatives
"E:\codes_python\relation latent world\E0\adp9\run_adp9_wm.cmd" adp9_b1
```

## 13. 最终研究判断

seed 0 的证据链已经闭合：

1. ADP6 形成统一 realization coordinate；
2. ADP8 证明 duplicates 可直接互相替代；
3. ADP9 learner-visible graph 自动找出 duplicates；
4. 冻结 mechanism 的逐步删除恢复恰好三个稳定 functional identities；
5. full TP support inference 在 IID 和 101 上通过。

下一项真正未解决的问题不再是 seed-0 algorithm design，而是 upstream multi-seed stability。只有生成独立的 ADP6-A2 seed 1/2 后，才能判断这条链是否具有 2/3 recovery rate。
