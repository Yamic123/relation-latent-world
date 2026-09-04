# E0-ADP7 实验报告：跨低层变化稳定性的功能合并

## 1. 最终结论

ADP7 按预注册 stage gate 执行到 A2，并在 cross-realization stability 阶段正式 **FAIL / STOP**。

实验确认了两件需要同时保留、不能互相替代的事实：

1. `C2+C5` 和 `C1+C4` 在 ADP6-A2 checkpoint 上确实是近乎完全相同的 functional duplicates。二者的跨 state prediction \(R^2\) 分别为 `0.999988` 和 `0.999970`。
2. 将每一对替换为一个单-candidate 容量模型后，source-copy 初始化能通过 IID 和双向 cross-state holdout，但在若干 cross-realization holdout 上超过容许的 `+0.03 NRMSE`。

因此本轮不能执行正式 merge：

\[
\boxed{
\text{checkpoint-level functional equivalence 已证实，}
\quad
\text{retraining 下的 realization stability 未证实。}
}
\]

A3、A4、A5 均未执行，不应记作 FAIL；它们是因 A2 gate 失败而合法停止。

## 2. 实验设置和复用约束

- 起点：ADP6-A2 `round_039.pt` 及 `round_039.npz` assignments。
- development seed：`0`。
- world/dataset/observable alpha/mechanism architecture 均保持不变。
- merge 数据使用 ADP6 没有用于 mechanism gradient update 的 validation families，避免把原训练 families 再随机切分造成明显泄漏。
- suspected pairs：`C2+C5`、`C1+C4`。
- wrong-pair audit：`C2+C3`、`C3+C4`、`C1+C2`。
- merged model：`SetMechanismBank(M=1, dim=64, heads=4)`，共 `54,850` 个参数。
- merged model 只有一个 adapter 和一套

  \[
  v=a_{jk}\alpha+b_{jk}.
  \]

- 没有拼接两个网络、增加 hidden dimension/heads/layers，或保留两套内部 coordinate。
- 优化：AdamW；mechanism LR `3e-4`；coordinate LR `1e-3`；weight decay `1e-5`；40 epochs；response batch 192；gradient clip 1.0。
- 两种初始化均运行：random 和 better-source candidate copy。
- source candidate 仅按 learner-visible training loss 选择，不读取 GT identity。
- GT identity 只用于 evaluator 指标和事后解释。

### 初始化 gate 约定

执行方案第 17 节要求两种初始化都运行，并说明两者都通过会提供更强证据；它没有将 random convergence 明列为 A0–A4 的硬阈值。本实现预先在配置中指定 `source_copy` 为正式 stage gate，random 作为独立 optimization robustness 指标。两条结果均完整保存，没有择优隐藏。

如果要求“两种初始化必须同时通过”，那么 ADP7 会更早在 A0 失败；这不会改变最终“不允许 merge”的结论。

## 3. ADP7-0：duplicate pair 审计

本阶段不训练。

| Pair | cross-state prediction \(R^2\) | 双向 cross-family \(R^2\) | \(|\Delta a|\) | \(|\Delta b|\) | state coverage overlap | evaluator GT 一致 |
|---|---:|---|---:|---:|---:|---|
| C2+C5 | 0.999988 | 0.999986 / 0.999990 | 0.000383 | 0.002149 | 0.7298 | 是 |
| C1+C4 | 0.999970 | 0.999970 / 0.999961 | 0.000618 | 0.000203 | 0.6999 | 是 |
| C2+C3 | -2.3461 | -2.7340 / -2.0093 | 0.000671 | 0.023112 | 0.7868 | 否 |
| C3+C4 | -2.2796 | -2.2352 / -3.9626 | 0.042967 | 0.097172 | 0.7463 | 否 |
| C1+C2 | -1.9525 | -2.0863 / -2.2265 | 0.044255 | 0.120081 | 0.8291 | 否 |

审计具有明显区分度：正确 pairs 的函数预测几乎相同，而错误 pairs 即使 state coverage overlap 较高，函数预测仍显著不一致。单看 \(a,b\) 距离不够，例如错误 pair `C2+C3` 的 \(|\Delta a|\) 也很小。

## 4. A0：IID heldout merge sanity

按 base ID 随机划分 70%/30%，任何 family 不跨 split。

| Pair | Init | separate NRMSE | merged NRMSE | merged functional \(R^2\) | 判定 |
|---|---|---:|---:|---:|---|
| C2+C5 | random | 0.00720 | 0.49174 | 0.82364 | FAIL（鲁棒性） |
| C2+C5 | source-copy | 0.00720 | **0.00648** | **0.99993** | PASS |
| C1+C4 | random | 0.00657 | 0.37176 | 0.86636 | FAIL（鲁棒性） |
| C1+C4 | source-copy | 0.00657 | **0.00526** | **0.99992** | PASS |

source-copy 同时满足：relative IID loss delta < 0.05、functional \(R^2>0.95\)、affine realization \(R^2>0.95\)。A0 primary gate PASS。

random 初始化在 40 epochs 内明显未收敛，说明“能否单模型表示”与“能否从随机初始化在短预算内重新学出”仍是两个问题。

## 5. A1：cross-state stability

| Pair | State split | separate NRMSE | merged NRMSE | \(\Delta_S\) | functional \(R^2\) | Primary |
|---|---|---:|---:|---:|---:|---|
| C2+C5 | middle → extreme | 0.00749 | 0.00663 | -0.00086 | 0.99992 | PASS |
| C2+C5 | extreme → middle | 0.00652 | 0.00878 | +0.00226 | 0.99991 | PASS |
| C1+C4 | middle → extreme | 0.00699 | 0.00636 | -0.00063 | 0.99991 | PASS |
| C1+C4 | extreme → middle | 0.00632 | 0.00694 | +0.00062 | 0.99990 | PASS |

所有 source-copy 结果均满足 \(\Delta_S<0.03\)、functional \(R^2>0.90\)、coordinate affine \(R^2>0.95\)。A1 primary gate PASS。

random 初始化四项均失败，merged NRMSE 为 `0.376–0.645`，继续表明当前 40-epoch 预算不具备 random-init robustness。

## 6. A2：cross-realization stability

### 6.1 Source-copy 正式结果

| Pair | Realization split | separate NRMSE | merged NRMSE | 差值 | functional \(R^2\) | 判定 |
|---|---|---:|---:|---:|---:|---|
| C2+C5 | weak → strong | 0.00634 | 0.03650 | **+0.03016** | 0.99980 | FAIL |
| C2+C5 | strong → weak | 0.01011 | 0.04843 | **+0.03832** | 0.99966 | FAIL |
| C2+C5 | negative → positive | 0.00704 | 0.02731 | +0.02027 | 0.99962 | PASS |
| C2+C5 | positive → negative | 0.00844 | 0.06195 | **+0.05351** | 0.99848 | FAIL |
| C1+C4 | weak → strong | 0.00554 | 0.01475 | +0.00921 | 0.99986 | PASS |
| C1+C4 | strong → weak | 0.01106 | 0.02348 | +0.01242 | 0.99988 | PASS |
| C1+C4 | negative → positive | 0.00751 | 0.05107 | **+0.04356** | 0.99895 | FAIL |
| C1+C4 | positive → negative | 0.00717 | 0.05102 | **+0.04385** | 0.99893 | FAIL |

预注册判据要求每一种 split 都满足：

```text
merged NRMSE <= separate NRMSE + 0.03
functional R² > 0.90
order accuracy > 0.95
affine realization R² > 0.95
```

所有 source-copy 行的 functional \(R^2\)、order accuracy 和 affine realization \(R^2\) 均通过；失败全部来自 held-out response NRMSE。

`C2+C5 weak→strong` 超阈值约 `0.00016`，单独看属于边界失败。但同一 pair 另有 `+0.03832` 和 `+0.05351`，`C1+C4` 也有两个约 `+0.044` 的失败，因此整体结论不是一个舍入误差造成的。

### 6.2 Random-init robustness

random 初始化的 merged NRMSE 为 `0.396–1.768`；单侧外推时 functional \(R^2\) 甚至可变为负数。random-init robustness 明确 FAIL。

## 7. 失败模式解释

这次失败不是以下几种情况：

- 不是两个 suspected candidates 本来对应不同 GT；audit 已排除。
- 不是单模型容量在 IID 下无法容纳两侧；A0 已通过。
- 不是 state coverage 分工；A1 双向均通过。
- 不是 coordinate 顺序或 affine gauge 崩溃；A2 中 order/affine 指标始终约为 1。
- 不是 functional probe 全面崩溃；source-copy functional \(R^2\) 始终为 `0.998–1.000`。

具体失败是：从一个已经正确的 source candidate 出发，只在受限 realization 区间继续拟合 merged union 后，模型在未见强度或另一符号侧产生了可测的 response drift。functional \(R^2\) 对整体函数形状仍很高，但不足以保证接近 separate baseline 的严格低 NRMSE。

因此应把结论表述为：

> 两个 candidates 在当前 checkpoint 上功能等价，但当前 merge retraining procedure 不具备足够的跨-realization 保真性。

这不等价于证明“数学上绝对不能 merge”；它证明的是本方案规定的、需要重新训练单模型的正式 merge 尚不安全。

## 8. 未执行阶段和 negative controls

| 阶段 | 状态 | 原因 |
|---|---|---|
| A3 cross-base | NOT RUN | A2 FAIL |
| A4 joint holdout | NOT RUN | A2 FAIL |
| A5 formal merge / \(\lambda_G\) sweep | NOT RUN | A2 FAIL |
| double-capacity merge | NOT RUN | A2 gate 后停止 |
| family-specific coordinate merge | NOT RUN | A2 gate 后停止 |
| wrong-pair trained merge | NOT RUN | A2 gate 后停止；但 ADP7-0 已完成三组 wrong-pair 无训练审计 |

未运行这些阶段符合“任一步 FAIL，STOP”的方案约束。尤其不能因为正确 pair 的 checkpoint similarity 很高，就跳过 A2 直接做 global count merge。

## 9. 计算代价

- 资源探针：1 epoch，CUDA peak allocation `21.42 MiB`。
- 所有正式训练行 CUDA peak allocation约 `21.42 MiB`。
- A0 最终正式运行：`6.47 s`。
- A1：`9.77 s`。
- A2：`18.41 s`。
- 单个 40-epoch fit：约 `0.67–1.17 s`。

3070 Ti 8 GB 无需缩小数据、模型或 batch；实验的主要成本是多 split/multi-init 重复训练，而非显存。

## 10. 输出结构

```text
E0/outputs/e0_adp7/
├── final_verdict.json
├── adp7_0_pair_audit/
│   ├── config.json
│   └── final_metrics.json
├── a0_iid_merge/
│   ├── config.json
│   └── final_metrics.json
├── a1_state_holdout/
│   ├── config.json
│   └── final_metrics.json
└── a2_realization_holdout/
    ├── config.json
    └── final_metrics.json
```

每个训练结果均保存完整 40-epoch loss trajectory、split 样本数、separate/merged MSE 与 NRMSE、functional \(R^2\)、coordinate 参数、affine/order 指标、参数量、runtime 和 CUDA peak allocation。

## 11. Reproducibility checklist

- [x] 使用固定 world、dataset 和 optimization seeds。
- [x] 保存 resolved config、脚本 SHA256 和 Git commit。
- [x] 保存 ADP6 输入 checkpoint 与 assignment 路径。
- [x] 按 base ID 切分，family 不跨 split 泄漏。
- [x] source selection 只读 learner-visible train loss。
- [x] GT 只用于 evaluator。
- [x] 两种初始化全部运行并分别报告。
- [x] merged model 为单-candidate 容量，未扩大 hidden/heads/layers。
- [x] 所有已获 gate 授权的 structured holdout 均按方案双向运行。
- [x] 使用 `wm` Conda DLL 安全启动方式。
- [x] A2 FAIL 后严格停止 A3/A4/A5。

复现命令：

```bat
"E:\codes_python\relation latent world\E0\adp7\run_adp7_wm.cmd" adp7_0 0
"E:\codes_python\relation latent world\E0\adp7\run_adp7_wm.cmd" adp7_a0 0
"E:\codes_python\relation latent world\E0\adp7\run_adp7_wm.cmd" adp7_a1 0
"E:\codes_python\relation latent world\E0\adp7\run_adp7_wm.cmd" adp7_a2 0
```

当前实现 SHA256：`ecea31f38f66b5b96a2fb6af23d3fee3e99b52b433ea2904d5b29e10ac8b709c`。

## 12. 下一步建议

下一步不应调松 A2 阈值后强行 merge。更有价值的诊断是固定 source mechanism，只比较以下三种操作：

1. 直接删除 duplicate、完全不 fine-tune；
2. merge 后只更新共享 affine \(a,b\)，冻结 mechanism；
3. 当前同时更新 mechanism 与 coordinate 的 merge retraining。

如果前两种能保持 cross-realization，而第三种失败，就能把问题定位为 merge fine-tuning 引起的 catastrophic drift，而不是高层 identity 不成立。
