# E0-ADP11 实验报告：Perfect-TP 条件下的 From-Zero 稳定性

## 1. 最终结论

ADP11 已完成预注册的双条件 3-seed pilot，并依据 gate 停止：

| 条件 | Z4 identity/coordinate PASS | End-to-end PASS | Pilot gate |
|---|---:|---:|---|
| Identity TP | 2/3 | 0/3 | FAIL |
| Orthogonal TP | 2/3 | 0/3 | FAIL |

因为两个条件都没有达到 end-to-end \(\ge2/3\)，实验没有扩展到 10 seeds，也没有在失败 gate 后继续运行 B1。

\[
\boxed{\textbf{ADP11 Pilot FAIL}}
\]

但失败并不主要发生在 mechanism formation：2/3 seeds 已从完全随机的 bank、assignment 和 coordinate parameters 恢复三个高质量 functional mechanisms。主要失败发生在 global count recovery：多余 candidates 在 from-zero hard assignment 中变成 unused 或低使用率、未充分成形的槽位，而不是 ADP10 中那种功能完整的 duplicates；只会删除 functional duplicates 的 ADP9 graph 无法清除这些槽位。

---

## 2. 实际执行配置

固定：

```text
WORLD_SEED       = 20260901
DATASET_SEED     = 20260902
FAMILY_SEED      = 20260905
Mmax             = 5
mechanism dim    = 64
mechanism heads  = 4
rounds           = 40
epochs / round   = 4
family batch     = 32
mechanism lr     = 3e-4
coordinate lr    = 1e-3
weight decay     = 1e-5
```

每个 condition-seed 均从以下状态开始：

- 随机初始化 5-candidate mechanism bank；
- random-balanced whole-family assignment；
- 随机初始化 candidate-level \(a,b\)；
- 不加载 ADP5/ADP6 mechanism checkpoint；
- 不加载 ADP9 graph，不硬编码 duplicate pair；
- 只复用 ADP5 的无泄漏 family 数据构造。

随机 coordinate 初始化在正式 pilot 前固定为：

```text
a_j = random sign × Uniform(0.8, 1.2)
b_j ~ Uniform(-0.1, 0.1)
```

“主要 candidate”在资源 probe 后、其余正式 pilot 开始前固定定义为 family usage ≥2%。最初 probe 结果已保存，没有覆盖。该定义只解释方案中未量化的“主要 purity-high candidate”，没有改变训练、graph、count 或 B1 gate。

---

## 3. Z0 独立性与 TP 审计

| 审计项 | 结果 |
|---|---:|
| 不同 optimization seeds | 3 |
| unique initial model hashes | 3 |
| unique initial assignment hashes | 3 |
| identity/orthogonal paired initial model 相同 | 3/3 |
| non-TP data/family hashes | 1 |
| TP data hashes | 2 |
| orthogonal \(B\) hashes | 1 |
| \(\max|B^TB-I|\) | \(1.19\times10^{-7}\) |

固定矩阵保存在：

```text
E0/outputs/e0_adp11_from_zero/orthogonal_tp/B.npy
```

文件 SHA-256：

```text
fbec97efa9b3f43df560a477ef21662a16319a08b67fba239bd589709d7a31c5
```

---

## 4. 一个重要的 TP 方案边界

执行前的数据流审计发现，ADP11 文档给出的正式方程中：

- Z2–Z4 family cost 使用 \(S,d,A,\alpha\)；
- mechanism forward 使用 \(S,v\)；
- Z8 直接使用 \(\Delta S\) 作为 reconstruction target；
- \(p^{id}\) 或 \(p^{orth}\) 没有进入 loss、assignment、coordinate 或 mechanism forward。

因此在不擅自加入 TP decoder、也不违反“禁止显式 inverse-\(B\)”的条件下，两种 TP 的优化问题完全相同。实测 paired-by-seed：

```text
final NRMSE delta                 = 0 for 3/3 pairs
matched functional R² delta      = 0 for 3/3 pairs
candidate family counts identical = 3/3 pairs
failure mode identical            = 3/3 pairs
```

所以：

\[
|\hat p_{id}-\hat p_{orth}|=0
\]

虽然数值满足差距 gate，但这是由实现路径中 `p` 未被消费保证的，不能作为非平凡的 TP representation invariance 证据。

---

## 5. Pilot 最终结果

Identity 与 Orthogonal 的 paired results 完全相同：

| TP | Seed | Z4 identity | Coord | Z4 | Final M | Prune | B1 | Final | Failure |
|---|---:|---|---|---|---:|---|---|---|---|
| Identity | 0 | PASS | PASS | PASS | 5 | FAIL | 未运行 | FAIL | F4 |
| Identity | 1 | FAIL | PASS | FAIL | — | 未进入 | 未运行 | FAIL | F3 |
| Identity | 2 | PASS | PASS | PASS | 5 | FAIL | 未运行 | FAIL | F4 |
| Orthogonal | 0 | PASS | PASS | PASS | 5 | FAIL | 未运行 | FAIL | F4 |
| Orthogonal | 1 | FAIL | PASS | FAIL | — | 未进入 | 未运行 | FAIL | F3 |
| Orthogonal | 2 | PASS | PASS | PASS | 5 | FAIL | 未运行 | FAIL | F4 |

这里的 F4 更准确地应细分为：

```text
F4b: unused / under-formed candidate cannot be removed by a duplicate-only graph
```

而不是“一个已经成形的 duplicate 被 graph 漏检”。

---

## 6. From-zero 逐轮形成过程

### Seed 0

seed 0 从明显 mixing 开始：round 0 各 candidate purity 约 0.39–0.60，matched functional \(R^2\) 仅 0.164/0.266/-0.014，held-out NRMSE 0.926。随后：

| 事件 | 首次达到的 round |
|---|---:|
| 所有主要 candidates purity >0.90 | 12 |
| 三个 matched functional \(R^2>0.95\) | 14 |
| held-out NRMSE <0.05 | 19 |

最终：

```text
matched functional R² = [0.999334, 0.999476, 0.999190]
held-out NRMSE         = 0.03242
assignment change     = 0.00263
family counts         = [502, 12, 530, 477, 0]
major mask (≥2%)      = [1, 0, 1, 1, 0]
```

即三个主要 candidates 已正确成形，但另有一个 12-family 的 under-trained candidate 和一个 dead candidate。

### Seed 1

最终三个 GT 都有可匹配候选，coordinate 与 NRMSE 通过，但一个占 193 families 的主要 candidate purity 仅 0.809，构成真实 mixing：

```text
matched functional R² = [0.987664, 0.999311, 0.984850]
held-out NRMSE         = 0.03940
family counts         = [479, 193, 347, 423, 79]
purity                = [1.000, 0.809, 1.000, 1.000, 1.000]
```

因此按 Z4 gate 记 F3，并停止。

### Seed 2

seed 2 最干净地恢复了三个 active mechanisms：

```text
matched functional R² = [0.999673, 0.999836, 0.999748]
held-out NRMSE         = 0.01911
assignment change     = 0.0
family counts         = [0, 530, 0, 502, 489]
```

两个 candidate 完全没有 family usage，但它们仍存在于 `alive` bank 中。

---

## 7. 为什么 Z4 成功后仍无法恢复 M=3

ADP9 的压缩算子只删除满足下列条件的 pair：

```text
cross-state R² > 0.995
cross-realization R² > 0.995
cross-base R² > 0.995
replacement max ΔNRMSE < 0.01
```

ADP10 的起点中，5 个 candidates 全部承担 families，冗余项已经学习成完整 duplicate，因此可以通过 functional equivalence 合并。

ADP11 from-zero 的 hard assignment 产生了另一种冗余：

```text
candidate 获得很少或零 responsibility
→ mechanism 没有足够 gradient
→ candidate 保持 under-trained / random
→ 它既不是有效机制，也不等价于某个有效机制
→ duplicate graph 正确地拒绝建边
→ alive bank 仍保持 5 个槽位
```

seed 0 最接近的 pair 的三域最小 \(R^2\) 只有 0.446；seed 2 的候选 pair 跨域 \(R^2\) 为负。因此不能通过放宽 0.995 阈值解决，也不应把这些 pair 强行 merge。

这说明当前链路缺少的是：

\[
\boxed{\text{inactive/unsupported candidate elimination}}
\]

而不是更宽松的 duplicate equivalence gate。

---

## 8. 为什么没有继续 10 seeds、B1 和 controls

执行方案规定：Identity 与 Orthogonal pilot 都达到 \(\ge2/3\) 才扩展至 10 seeds。实测两者均为 0/3 end-to-end，因此：

- 没有运行 seeds 3–9；
- 没有对 F3/F4 seeds 事后增加 rounds；
- 没有放宽 duplicate threshold；
- 没有使用 \(M_{real}=3\) 强制删除；
- 没有越过失败的 pruning/count gate运行正式 B1；
- 没有启动依赖完整主链成功的昂贵 negative-control sweep。

这些缺失值表示“由 gate 截停”，不是计算失败或遗漏。

---

## 9. 计算代价

环境：conda `wm`，NVIDIA RTX 3070 Ti Laptop GPU 8GB。

| Condition | Seed | Z2–Z4 wall time | peak allocated GPU memory |
|---|---:|---:|---:|
| Identity | 0 | 217.8 s | 48.4 MiB |
| Identity | 1 | 194.8 s | 48.4 MiB |
| Identity | 2 | 194.7 s | 48.4 MiB |
| Orthogonal | 0 | 194.6 s | 48.4 MiB |
| Orthogonal | 1 | 145.5 s | 48.4 MiB |
| Orthogonal | 2 | 169.0 s | 48.4 MiB |

部分 runs 并行执行。全部 pilot、审计与汇总的实际经过时间约 18.5 分钟。没有因 8GB 显存限制缩小正式训练规模。

---

## 10. 输出与复现

实现：

```text
E0/adp11/run_adp11.py
E0/configs/e0_adp11.json
```

输出：

```text
E0/outputs/e0_adp11_from_zero/
├── identity_tp/seed_000..002/
├── orthogonal_tp/
│   ├── B.npy
│   └── seed_000..002/
└── aggregate_3seed/
    ├── summary.csv
    └── final_verdict.json
```

复现命令：

```powershell
conda run -n wm python E0/adp11/run_adp11.py --condition identity --seed 0
conda run -n wm python E0/adp11/run_adp11.py --condition orthogonal --seed 0
conda run -n wm python E0/adp11/run_adp11.py --stage aggregate --seeds 0,1,2
```

---

## 11. 科学结论与下一步

ADP11 对当前实现支持以下结论：

1. random-balanced whole-family alternating optimization 并非完全无效：2/3 seeds 能从零形成三个高质量 functional identities 和 shared realization coordinates；
2. 它不能稳定处理 overcomplete bank 的剩余槽位；hard assignment 使冗余 candidate 更可能饿死，而不是形成可被 equivalence graph 合并的 duplicate；
3. 当前 global count recovery 需要显式、learner-visible、带验证保护的 inactive-candidate pruning，且不能简单把“usage 低”直接等同于应删除；
4. identity/orthogonal 的当前对照没有进入优化方程，不能据此声称非平凡的 TP representation invariance。

建议下一步拆成两个独立问题：

- **ADP12-A：inactive-candidate pruning**——用跨轮低 usage、leave-one-out/reassignment safety 和 structured validation 联合判定能否删除 unsupported slots；
- **ADP12-B：non-vacuous TP invariance**——先明确一个不使用 GT 或显式 \(B^{-1}\)、但确实消费 \(p\) 的 learner module/loss，再比较 identity 与 orthogonal。

不要直接放宽 duplicate graph 的 \(R^2\) gate：本次多余 candidates 并不是近似 duplicate。
