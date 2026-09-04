# E0-ADP8 实验报告：merge identity 与 fine-tuning drift 的区分

## 1. 结论

ADP8 得到了清晰的正结果：

\[
\boxed{
\text{duplicate identity 本身成立；ADP7 的失败来自 merge 后联合 fine-tuning 的 catastrophic drift。}
}
\]

在 `C2+C5`、`C1+C4` 两个 pair 和四种 cross-realization split 上：

- 直接删除 duplicate、完全不训练：两个可能的 survivor 全部通过，共 `16/16`。
- 冻结 mechanism、只优化共享 \(a,b\)：全部通过，共 `8/8`。
- 同时更新 mechanism 与 coordinate：只有 `3/8` 通过，`5/8` 失败。

直接删除时，相对原 separate baseline 的最大 NRMSE 增量仅为 `0.001296`；coordinate-only 的最大增量仅为 `0.000581`。联合 fine-tuning 的最大增量为 `0.053509`。

因此不能再把 ADP7-A2 的失败解释为“两副本不是真正相同的 \(Z_D\)”。更准确的解释是：

> 两个副本在删除其中一个后已经可以直接互相替代；对已正确的 mechanism 继续施加受限 realization 数据上的 gradient，反而破坏了未见 realization 区域。

## 2. 实验控制

- 完全复用 ADP7 的 validation families、ADP6-A2 `round_039` checkpoint 和 assignments。
- 完全复用 ADP7 的四种 cross-realization split：
  - weak → strong；
  - strong → weak；
  - negative → positive；
  - positive → negative。
- separate baseline、NRMSE 定义和 PASS 阈值与 ADP7-A2 相同。
- suspected pairs：`C2+C5`、`C1+C4`。
- GT 仅用于 functional/affine evaluator，不参与 survivor 选择或训练。
- survivor 由 learner-visible training response loss 选择；直接删除操作额外报告两个 survivor 方向，防止选择偏差。
- coordinate-only 使用 40 epochs、Adam、LR `1e-3`，mechanism 全部 `requires_grad=False`。
- joint baseline 不重跑，直接复用 ADP7-A2 source-copy 结果。

每行 PASS 要求：

```text
merged NRMSE <= separate NRMSE + 0.03
functional R² > 0.90
affine realization R² > 0.95
order accuracy > 0.95
```

## 3. 操作一：直接删除 duplicate，不训练

下表给出两个 survivor 中较差方向的结果；即使保留较差副本也全部通过。

| Pair | Split | separate NRMSE | 较差 survivor NRMSE | 最大增量 | min functional \(R^2\) | PASS |
|---|---|---:|---:|---:|---:|---|
| C2+C5 | weak → strong | 0.006345 | 0.006683 | +0.000339 | 0.999925 | 是 |
| C2+C5 | strong → weak | 0.010112 | 0.010370 | +0.000258 | 0.999925 | 是 |
| C2+C5 | negative → positive | 0.007041 | 0.007522 | +0.000481 | 0.999925 | 是 |
| C2+C5 | positive → negative | 0.008441 | 0.008668 | +0.000227 | 0.999925 | 是 |
| C1+C4 | weak → strong | 0.005545 | 0.006502 | +0.000957 | 0.999883 | 是 |
| C1+C4 | strong → weak | 0.011061 | 0.012267 | +0.001206 | 0.999883 | 是 |
| C1+C4 | negative → positive | 0.007505 | 0.008624 | +0.001118 | 0.999883 | 是 |
| C1+C4 | positive → negative | 0.007170 | 0.008467 | +0.001296 | 0.999883 | 是 |

这项结果直接回答了最关键的问题：C2 本身能解释原来分给 C5 的 families，C5 也能解释 C2 的 families；C1/C4 同理。其误差变化远小于 `+0.03` 容忍度。

## 4. 操作二：冻结 mechanism，只调整共享 \(a,b\)

| Pair | Split | separate NRMSE | coordinate-only NRMSE | 增量 | min functional \(R^2\) | PASS |
|---|---|---:|---:|---:|---:|---|
| C2+C5 | weak → strong | 0.006345 | 0.006327 | -0.000018 | 0.999922 | 是 |
| C2+C5 | strong → weak | 0.010112 | 0.010216 | +0.000105 | 0.999919 | 是 |
| C2+C5 | negative → positive | 0.007041 | 0.007623 | +0.000581 | 0.999920 | 是 |
| C2+C5 | positive → negative | 0.008441 | 0.008948 | +0.000507 | 0.999899 | 是 |
| C1+C4 | weak → strong | 0.005545 | 0.005652 | +0.000107 | 0.999891 | 是 |
| C1+C4 | strong → weak | 0.011061 | 0.010935 | -0.000125 | 0.999890 | 是 |
| C1+C4 | negative → positive | 0.007505 | 0.007830 | +0.000325 | 0.999895 | 是 |
| C1+C4 | positive → negative | 0.007170 | 0.007349 | +0.000179 | 0.999886 | 是 |

只调整 coordinate 没有产生 extrapolation drift。最终 \(a,b\) 只在原 source 参数附近小幅移动，说明 coordinate adapter 不是 ADP7 失败的来源。

## 5. 操作三：mechanism + coordinate 联合 fine-tuning

| Pair | Split | separate NRMSE | joint NRMSE | 增量 | PASS |
|---|---|---:|---:|---:|---|
| C2+C5 | weak → strong | 0.006345 | 0.036503 | **+0.030158** | 否 |
| C2+C5 | strong → weak | 0.010112 | 0.048434 | **+0.038322** | 否 |
| C2+C5 | negative → positive | 0.007041 | 0.027307 | +0.020266 | 是 |
| C2+C5 | positive → negative | 0.008441 | 0.061951 | **+0.053509** | 否 |
| C1+C4 | weak → strong | 0.005545 | 0.014753 | +0.009208 | 是 |
| C1+C4 | strong → weak | 0.011061 | 0.023484 | +0.012423 | 是 |
| C1+C4 | negative → positive | 0.007505 | 0.051069 | **+0.043564** | 否 |
| C1+C4 | positive → negative | 0.007170 | 0.051016 | **+0.043845** | 否 |

联合更新 mechanism 后才出现系统性退化。由于 direct-delete 和 coordinate-only 使用完全相同的 source functions、families 和 split，却不出现退化，所以差异可归因于 mechanism gradient update。

## 6. 判定逻辑

实验预先使用以下逻辑：

```text
identity_supported = direct-delete 全通过 OR coordinate-only 全通过
catastrophic_drift = identity_supported AND joint fine-tuning 未全通过
```

实际结果：

```text
direct-delete best survivor: 8/8 PASS
direct-delete both survivor directions: 16/16 PASS
coordinate-only: 8/8 PASS
joint fine-tuning: 3/8 PASS

duplicate_identity_supported = true
catastrophic_drift_supported = true
verdict = catastrophic_drift_supported
```

## 7. 计算代价与输出

- 总 wall-clock：`15.10 s`。
- CUDA peak allocation：`45.74 MiB`。
- 没有重新训练 mechanism；唯一新增训练是八组两参数 \(a,b\) 优化。

输出：

```text
E0/outputs/e0_adp8/
├── config.json
└── final_metrics.json
```

`final_metrics.json` 包含：

- 16 行 direct-delete 结果；
- 8 行 best-survivor direct-delete 结果；
- 8 行 coordinate-only 结果及完整 40-epoch loss；
- 8 行复用的 ADP7 joint baseline；
- 汇总 checks 和最终 verdict。

## 8. Reproducibility checklist

- [x] 使用 ADP7 相同数据、assignment、split 和 threshold。
- [x] 两个 survivor 方向均测试。
- [x] survivor 选择不读取 GT。
- [x] coordinate-only 确认冻结全部 mechanism 参数。
- [x] joint baseline 原样复用，没有重新抽样或重跑。
- [x] 保存环境、配置、Git commit 和文件哈希。
- [x] `wm` CUDA/DLL 启动路径验证通过。

复现命令：

```bat
"E:\codes_python\relation latent world\E0\adp8\run_adp8_wm.cmd"
```

## 9. 后续含义

ADP7 原本计划“训练一个全新 merged mechanism”再决定是否合并，但 ADP8 表明当前 duplicates 已经无需重新训练：直接选一个 survivor 即可覆盖 union families。

因此下一步更合理的 merge 操作是：

1. 根据 learner-visible cross-family prediction equivalence 选择 survivor；
2. 将另一 candidate 的 assignments 重定向到 survivor；
3. 删除 duplicate；
4. 冻结 survivor mechanism，最多只校准共享 \(a,b\)；
5. 完成 cross-state、cross-base 和 joint holdout 后再恢复其他训练。

不应在 merge 时继续对 survivor mechanism 做 unrestricted fine-tuning。
