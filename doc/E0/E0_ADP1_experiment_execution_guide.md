# E0-ADP1 单卡最小验证方案（8GB RTX 3070 Ti）

> **当前执行版本**  
> 本方案专门面向一张 8GB RTX 3070 Ti。目标是用最小实验回答 ADP1 的核心科学问题，而不是复刻全规模 E0。  
> 原 10-seed、完整 100k 数据方案已移至 `E0_ADP1_full_scale_reference.md`，仅供未来扩展，不应在本轮执行。

---

## 0. 最小结论问题

E0-ADP1 只验证：

\[
\boxed{
\text{切断 assignment 与 mechanism 的同步 co-adaptation 后，}
\text{模型能否从随机初始化进入 O5 已证明存在的 GT-like basin？}
}
\]

为回答这个问题，本轮只做：

1. 固定 mechanisms，对每个 sample 枚举五个 candidate 的全部 \(2^5=32\) 个 support，并优化 active \(v\)；
2. 固定 \(m,v\)，用 residual/backfitting 更新 mechanisms；
3. 与同数据、同随机初始化、同更新预算的 joint baseline 配对比较；
4. 使用 3 个 optimization seeds 验证结果不是单 seed 偶然现象。

本轮**不训练 \(q_\eta\)**。原因是 \(q_\eta\) 只负责后续 amortization，不参与当前“能否发现正确 basin”的主问题。只有 ADP1-MINI Discovery PASS 后，才另开一个便宜的 amortization pilot。

必须明确：EM-style alternating optimization 不保证全局最优。“Exact”仅指 32 个离散 support 全部枚举；active \(v\) 的连续优化仍是多起点数值近似。

---

## 1. 为什么这个规模已经足以验证主假设

原 E0 已经完成并确认：

| 事实 | 已知证据 |
|---|---|
| mechanism architecture 能表达 GT | O1 PASS，IID NRMSE 约 0.007–0.009 |
| \(q_\eta\) capacity 足够 | O2 PASS |
| 正确 mechanism basis 下 objective 选择 GT support | O3-E top-1 recovery 0.997 |
| 原 joint 随机优化失败 | identity/orthogonal 均 0/10 PASS，五个 candidate 全开 |
| GT-like basin 存在且稳定 | O5 PASS |

因此 ADP1 不需要重新验证大样本 scaling，也不需要 relation、TP discovery 或 10-seed 稳健性宣称。一个固定小数据集、3 个 paired seeds 已足以完成 **proof-of-mechanism**：alternating 是否能把解从 dense basin 推向 GT-like basin。

若小实验成功，结论是“该优化机制可行，值得扩大”；不是“已经稳定解决 E0”。若小实验失败，也不能直接否定全规模 ADP1，只能说明最小设置下未观察到预期 basin transition。

---

## 2. 保持不变的科学控制

### 2.1 复用原 synthetic world

禁止重新生成数据。必须复用：

```text
E0/datasets/e0a_world_seed_20260901/
```

固定：

```text
WORLD_SEED   = 20260901
DATASET_SEED = 20260902
M_star       = 3
M_max        = 5
K            = 3 slots
d_s          = 2
d_v          = 1
d_p          = 6
lambda_P     = 1e-3
```

真实 world：

\[
\Delta S=e_1^*+e_2^*+e_3^*,
\]

\[
e_j^{k*}
=m_j^*C_{jk}
\left[v_j^*A_{jk}s^k+v_j^*b_{jk}+0.25(v_j^*)^2c_{jk}\right],
\]

\[
C=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&0&1
\end{bmatrix}.
\]

训练中不出现 `101`；`test_combination_101` 全部为 `101`。

### 2.2 复用原 model

必须复用原 `SetMechanismBank`：

```text
token dim = 64
attention heads = 4
M_max = 5
state encoder + [v,v²] encoder + FiLM + set attention + candidate adapters
```

禁止改成五套独立 MLP、GT 方程、线性字典或已知 incidence matrix。

### 2.3 复用原 optimizer 与 loss

```text
M-step optimizer = AdamW
learning rate     = 3e-4
weight decay      = 1e-5
gradient clip     = 1.0
effect loss       = MSE，按 3×2 coordinates 取 mean
lambda_P          = 1e-3
```

禁止再次 sweep \(\lambda_P\)。

### 2.4 本轮唯一变化

| Joint baseline | ADP1-MINI |
|---|---|
| \(q_\eta\)、gates、\(v\)、mechanisms 同步梯度更新 | E-step 与 M-step 互相冻结 |
| Hard-Concrete relaxed gate | 枚举全部 32 个 binary support |
| \(q_\eta\) 直接预测 \(v\) | 每个 sample/support 局部优化 active \(v\) |
| 五个 mechanisms 同时协作 | residual/backfitting 逐 candidate 更新 |

不加入 relation、soft EM、anchor、curriculum、额外结构 loss、candidate pruning/death、GT initialization 或 O1/O5 checkpoint。

### 2.5 TP 在 MINI 主实验中的处理

ADP1 discovery 的 local solver读取 \((S,\Delta S)\)，不调用 \(q_\eta(S,p)\)，因此 identity TP 与 orthogonal TP 在这一阶段会产生完全相同的计算。为避免把主实验无意义地重复两遍：

- MINI discovery 每 seed 只运行一次；
- paired joint mini control 只运行 identity TP；
- 原数据中的 identity/orthogonal TP 均保留不改；
- 只有可选 amortization pilot 才分别测试 identity 与 orthogonal。

这不是删除 TP 条件，而是避免重复一段数学上不依赖 TP 的 discovery 计算。

---

## 3. 小数据子集

### 3.1 子集规模

从现有 split 取固定前缀，不复制或重新生成 world：

| split | 原规模 | MINI 使用量 | 用途 |
|---|---:|---:|---|
| train | 100,000 | **1,024** | E/M-step |
| val | 20,000 | **256** | checkpoint 与稳定性 |
| test_iid | 20,000 | **512** | 主结构评估 |
| test_context | 20,000 | **512** | context 泛化 |
| test_combination_101 | 20,000 | **512** | 组合泛化 |

固定索引：

```python
train_idx = np.arange(1024)
val_idx = np.arange(256)
test_iid_idx = np.arange(512)
test_context_idx = np.arange(512)
test_101_idx = np.arange(512)
```

所有 optimization seeds 使用完全相同的索引。不能为每个 seed 重新抽样，也不能根据 GT labels 选择“更容易”的 sample。

### 3.2 覆盖检查

训练程序只读取 `visible.npz` 的 `S` 与 `delta_S`。单独的 evaluation 程序可在子集索引冻结后读取 `hidden_gt.npz`，确认六个非零训练 pattern 均有至少 100 个样本、`000` 至少 20 个样本。

若固定前 1,024 个样本意外不满足覆盖条件，实验标记 `INVALID_SUBSET`；此时可把 **统一前缀** 增至 2,048，但不得按 GT pattern 手工挑样本。该扩容必须对三个 seeds 和 paired baseline 同时生效。

### 3.3 输入有效性

运行前必须确认：

```text
oracle MSE < 1e-10
train 中 101 数量 = 0
manifest SHA-256 全部匹配
```

任一不满足则停止，不计为 ADP1 FAIL。

---

## 4. 8GB 显存安全实现

显存约束的关键不是减少 32 个 support，而是**串流枚举**。

### 4.1 禁止同时展开全部搜索维度

不得构造：

```text
[sample_batch, 32 supports, 3 restarts, ...]
```

推荐显存中任一时刻只存在：

```text
[32 samples, 8 supports, 1 restart, 5 candidates, 3 slots, hidden_dim=64]
```

### 4.2 固定内存参数

```text
E-step sample chunk       = 32
support parallelism       = 8
restart parallelism       = 1
M-step batch size         = 128
dtype                     = float32
AMP                       = off for E-step
gradient checkpointing    = unnecessary
```

流程：

1. `S`、`delta_S`、当前 best score、best support、best \(v\) 常驻 CPU；
2. 每次只把 32 个 samples 放到 GPU；
3. 对 support code 0–31 分为四组、每组 8 个并行处理；
4. 每个 support 的 restart 顺序处理；
5. 每次 inner step 后立即释放计算图；
6. 当前 support 完成后，只把逐 sample 最优 score/\(v\) 与全局 best 比较；
7. 不保存完整 `[N,32,restart,...]` tensor。

要求在 smoke test 中记录 `torch.cuda.max_memory_allocated()`。峰值超过 6GB 时，依次把 support parallelism 从 8 降至 4、再把 sample chunk 从 32 降至 16；不得减少 support 数量。若显存余量充足，可把 support parallelism 提到 16 以缩短时间，但必须先通过数值一致性测试。

---

## 5. 最小实验分阶段执行

### Stage 0：16-sample smoke test

目的仅为发现实现错误，不作科学结论。

```text
train samples = 16
EM rounds = 2
seed = 0
```

必须验证：

- 32 个 support 全部被访问；
- E-step 前后 mechanism 参数 hash 不变；
- M-step 前后 stored \(m,v\) hash 不变；
- 无 NaN/Inf；
- peak GPU memory < 6GB；
- `00000` prediction 精确为 0；
- 两个 effect MSE 相同时，少一个 active bit 的 objective 少 \(10^{-3}\)。

Stage 0 通过后才能运行 Stage 1。

### Stage 1：单 seed proof-of-mechanism

```text
seed = 0
train = 1024
val = 256
max EM rounds = 12
```

Stage 1 不允许据结果修改 \(\lambda_P\)、model architecture、support 集合或 Pass 阈值。只允许修复 implementation bug。

### Stage 2：最小复现

Stage 1 有效完成后运行：

```text
seeds = 0,1,2
```

无论 seed 0 成功或失败，最终判断都以三 seed 结果为准。seed 0 不是筛选门。

### Stage 3：paired joint baseline

在同一 1,024/256 子集、相同 seeds、相同 initial full-model state 和相同 960 optimizer-update budget下运行 **identity TP joint baseline**。Joint 运行很便宜，不应省略；orthogonal joint mini 不属于最小主实验。

### Stage 4：可选 amortization pilot

仅当 Stage 2 判为 `ADP1-MINI PASS` 时，另用一个 discovery 成功 seed 做：

```text
q_eta identity TP, max 30 epochs
q_eta orthogonal TP, max 30 epochs
```

该结果只回答 solver 能否被蒸馏，不进入 ADP1-MINI Discovery 主判定。若显存或时间紧张，可完全延期。

---

## 6. E-step：完整 32-support hard assignment

冻结 mechanism bank \(\Theta^{(r)}\)。对每个 sample \(i\) 与：

\[
s\in\{0,1\}^5
\]

求：

\[
v_{i,s}^*
=
\arg\min_v
\operatorname{MSE}
\left(
\Delta S_i,
\sum_{j=1}^{5}s_j\mathcal M_j(S_i,v_j;\Theta^{(r)})
\right),
\]

其中 active \(v_j\in[-1.5,1.5]\)，inactive \(v_j=0\)。随后计算：

\[
J_i(s)=L_{effect,i}(s,v_{i,s}^*)+10^{-3}|s|.
\]

选择最小 \(J_i(s)\) 的 support。

必须包含 `00000` 与 `11111`；不得设置 `max_active`、beam search、top-k support 或 q proposal。

### 6.1 小型 continuous solver

先从最便宜配置开始：

```text
v = 1.5 * tanh(z)
optimizer = Adam
inner LR = 0.05
inner steps = 20
restarts = 2
```

两个起点：

1. \(v=0\)；
2. 上一 round 的 selected \(v\) 投影到当前 support；
第 2 项在 round 0 没有历史 selected \(v\) 时，改为由 `seed, sample_index, support_code` 唯一决定的随机起点。

在 16 个固定 samples 上，用 10-restart L-BFGS-B 复核。按以下预注册数值精度阶梯选择**最小达标配置**：

```text
A: 20 steps, 2 restarts
B: 30 steps, 2 restarts
C: 30 steps, 3 restarts
```

达标要求：

```text
selected-support agreement >= 99%
median objective gap <= 1e-5
```

一旦选定 A/B/C，三个正式 seeds 必须使用同一配置。该阶梯只校准 continuous solver 数值精度，不是对科学结果调参。若 C 仍不达标，停止并修 solver。

### 6.2 Tie-break

若 objective 差小于 \(10^{-8}\)：

1. 选择 active count 更小者；
2. 再相同则选择 bit code 更小者。

---

## 7. M-step：固定 assignment 的 residual/backfitting

完成全 train E-step 后冻结 \(m,v\)。对 candidate \(j\)：

\[
r_i^j
=
\Delta S_i-
\operatorname{stopgrad}
\left(
\sum_{k\ne j}m_i^k\mathcal M_k(S_i,v_i^k)
\right).
\]

只在 \(m_i^j=1\) 的 samples 上优化：

\[
L_j=\operatorname{MSE}(\mathcal M_j(S_i,v_i^j),r_i^j).
\]

最小预算：

```text
max EM rounds = 12
updates per candidate per round = 16
M-step batch size = 128
total maximum updates = 12 × 5 × 16 = 960
```

每个 minibatch update 前用最新 bank 重算 residual。candidate 顺序按 `seed + round` 生成并记录。

原 bank 有共享 backbone。更新 candidate \(j\) 会更新 shared parameters，并可能间接改变其他 candidates；这是原 architecture 的一部分，不能复制五套 backbone。必须保证其他 candidate adapter rows 不变。由于 adapter 是单一 `[5,dim]` Parameter，AdamW 即使面对零 gradient 也可能通过 weight decay/moments 改变其他行，因此必须 snapshot/restore非 \(j\) adapter rows 及相应 optimizer moments，不能只做 gradient mask。

若某 candidate 当轮无 active samples，跳过其 M-step，但下一 E-step 仍枚举包含它的 support；不得 prune、reinitialize 或 rescue。

---

## 8. Outer-loop 稳定与 checkpoint

每轮对 train 重新 E-step；val 每 2 rounds 做一次完整 E-step，以节省时间。

记录：

\[
\bar J,
\quad
\bar L_{effect},
\quad
\overline{|m|},
\quad
u_1,\ldots,u_5,
\]

以及 support change rate：

\[
SCR^{(r)}
=
\frac1N\sum_i\mathbf1[m_i^{(r)}\ne m_i^{(r-1)}].
\]

保存 val penalized objective 最低的 bank。稳定条件：

```text
round >= 4
train SCR < 0.01，连续 3 rounds
最近两次 val objective 相对变化 < 5e-4
```

达到稳定可提前停止；否则最多 12 rounds。恢复 best bank 后必须重新对 train/val 完整 E-step，固化最终 assignments。

hidden GT 不能用于 checkpoint、early stopping 或数值 solver 配置选择。

---

## 9. 伪代码

```python
for seed in [0, 1, 2]:
    full_model = build_original_e0a_model(seed)
    save_initial_full_state(full_model)
    bank = copy(full_model.bank)
    discard_q_eta_for_discovery()

    best_val_J = inf
    previous_m = None

    for round_id in range(12):
        freeze(bank)
        latent = stream_exact_e_step(
            bank, train_prefix_1024,
            supports=range(32),
            sample_chunk=32,
            support_parallelism=8,
            restart_parallelism=1,
            v_solver=frozen_solver_config,
            lambda_p=1e-3,
        )

        scr = mean(any(latent.m != previous_m, axis=1))

        if round_id % 2 == 0:
            val_latent = stream_exact_e_step(bank, val_prefix_256)
            maybe_save_best_by_val_J(bank, val_latent)

        unfreeze(bank)
        freeze(latent.m, latent.v)
        for j in seeded_candidate_order(seed, round_id):
            active = where(latent.m[:, j] == 1)
            for step in range(16):
                batch = sample(active, 128, replacement=True)
                other = current_other_effects(bank, batch, j, latent)
                residual = stopgrad(delta[batch] - other)
                loss = mse(bank.effect(j, S[batch], latent.v[batch, j]), residual)
                row_isolated_adamw_step(loss)

        previous_m = latent.m.copy()
        if stable_for_three_rounds() and round_id >= 6:
            break

    restore_best_bank()
    final_train_latent = stream_exact_e_step(bank, train_prefix_1024)
    evaluate_on_fixed_small_tests(bank, seed)

    # Cheap paired control from exactly the same saved initial full state.
    run_joint_baseline(
        initial_state=saved_initial_full_state,
        train=train_prefix_1024,
        val=val_prefix_256,
        optimizer_updates=960,
    )
```

---

## 10. 指标

### 10.1 必须报告

| 指标 | 含义 |
|---|---|
| IID/context/101 NRMSE | prediction 与组合泛化 |
| participation F1 | assignment 是否对应 GT |
| exact pattern accuracy | 每 sample 的整体 support 是否正确 |
| instance-effect \(R^2\) | 每 candidate 在 interaction 上的贡献是否对应 GT mechanism |
| functional \(R^2\) | 是否学到可复用 mechanism function |
| candidate usage \(u_j\) | 是否 all-on/all-off |
| redundant usage max | 两个多余 candidates 是否关闭 |
| support-change rate | assignment 是否稳定 |
| support objective margin | 最优 support 是否脆弱 |
| leave-one-out gain \(G_j\) | 是否仍为五候选 distributed code |
| peak VRAM / wall time | 单卡可执行性 |

NRMSE：

\[
NRMSE
=
\frac{\sqrt{E\|\Delta S-\widehat{\Delta S}\|^2}}
{\sqrt{E\|\Delta S-E[\Delta S]\|^2}}.
\]

Instance effect：

\[
\hat e_i^l=m_i^l\mathcal M_l(S_i,v_i^l),
\qquad
R^2_{instance}(l,j)=R^2(\hat e^l,e^{j*}).
\]

先按原 E0 functional \(R^2\) matrix 做 Hungarian permutation alignment，再报告 F1、instance-effect 与两个 redundant candidates。

`test_combination_101` 的 GT \(Z_2\) 恒 inactive，其 per-mechanism F1/R² 为常数退化指标，应写 N/A；该 split 主要报告 NRMSE 与 `101` pattern exact accuracy。

### 10.2 小样本置信度

对 test-IID 的 512 samples 做 sample bootstrap 1,000 次，报告 NRMSE、F1、instance \(R^2\) 的 95% CI。跨 optimization seed 只报告三个原始值、mean 和 range；三 seed太少，不伪造稳定的 seed-level CI。

---

## 11. ADP1-MINI Pass / Fail

### 11.1 单 seed PASS

必须同时满足：

```text
discovery stable = true
IID NRMSE < 0.05
101 NRMSE < 0.10
participation F1 > 0.90
instance-effect R² min > 0.90
functional R² min > 0.90
redundant usage max < 0.10
```

同时报告更严格的 `101 NRMSE < 0.05` 是否达到，但 MINI 主判定保留原 E0 compositional minimum `<0.10`，避免小数据训练被过度苛刻的 secondary threshold 否决。

### 11.2 三 seed 总判定

| 判定 | 标准 | 可得结论 |
|---|---|---|
| `ADP1-MINI PASS` | **3/3** seeds 单 seed PASS | alternating 机制在小型受控实验中可重复进入 GT-like basin，值得扩大 |
| `ADP1-MINI PARTIAL` | **2/3** seeds PASS | 存在明显信号，但 basin 吸引域仍不稳定 |
| `ADP1-MINI FAIL` | **0/3 或 1/3** seeds PASS | 最小 alternating 干预未提供可重复恢复 |

这不是原 E0 的“8/10 稳定恢复”认证。只有未来扩大到 10 seeds、full data 且至少 8/10 PASS，才能声称可靠恢复。

### 11.3 必须优于 paired joint baseline

即使 ADP1 达到自身阈值，也必须看到相同三个 seeds 上：

```text
ADP1 structural recovery 明显高于 joint
ADP1 redundant usage 明显低于 joint
```

如果 joint mini baseline 同样 3/3 PASS，则小数据条件没有复现原 failure mode，ADP1 的因果对照失效，标记 `MINI_CONTROL_NOT_DIAGNOSTIC`；此时不能把 ADP1 成功归因于 alternating。

---

## 12. 失败模式解释

| 结果 | 含义 |
|---|---|
| 全部迅速选 `00000` | random mechanisms 初期不足以抵消 sparsity penalty；hard assignment all-zero collapse |
| 长期接近 `11111`，五个 leave-one-out gain 均正 | 即使 exact binary sparsity 仍形成 dense cooperative basin |
| 只有 1–2 个 candidate active，NRMSE 高 | early monopoly / mechanism merge，不是成功稀疏 |
| active count 约 3，但 functional/instance R² 低 | 任意 mixed three-basis，不是 GT ontology |
| NRMSE 低但 F1/R² 低 | local \(v\) 或 mixed mechanism predictive shortcut |
| assignment 在 patterns 间振荡 | hard alternating 不稳定，或 continuous solver margin 太小 |
| objective 连续两轮增加 >1% | residual/M-step 或 inner solver 数值实现警报，先排错 |
| ADP1 仅 seed 0 成功 | 偶然进入 basin，不构成可重复验证 |
| ADP1 3/3 成功、joint 0/3 失败 | 对主假设最干净的支持 |
| ADP1 与 joint 都成功 | mini control 未复现原问题，不能隔离 alternating 的作用 |
| ADP1 与 joint 都失败 | 小样本容量或优化预算可能不足；本轮结论仅为 MINI FAIL |

ADP1-MINI FAIL 不等于 GT-like basin 不存在；O5 已证明其存在。它只说明当前最小 hard alternating solver 未能从随机初始化可重复到达该 basin。

---

## 13. 输出结构

```text
E0/outputs/e0_adp1_mini/
├── input_validation.json
├── subset_indices.npz
├── solver_sanity/
│   ├── selected_config.json
│   └── metrics.json
├── seed_0/
│   ├── run_metadata.json
│   ├── initial_full_model.pt
│   ├── initial_bank.sha256
│   ├── adp1/
│   │   ├── best_bank.pt
│   │   ├── final_assignments.npz
│   │   ├── outer_curves.json
│   │   ├── metrics.json
│   │   └── stdout.log
│   ├── joint_mini/
│   │   ├── best_checkpoint.pt
│   │   ├── training_curves.json
│   │   └── metrics.json
│   └── figures/
├── seed_1/
├── seed_2/
├── paired_comparison.csv
├── summary.json
├── E0_ADP1_MINI_report.md
└── reproducibility_checklist.md
```

每个 run 记录 Python/PyTorch/CUDA、GPU、peak VRAM、wall time、git commit、dirty state、全部 seeds、dataset hashes、initial state hash、solver A/B/C 配置与实际 optimizer update 数。

---

## 14. 8GB 默认配置

```json
{
  "experiment": "E0-ADP1-MINI",
  "optimization_seeds": [0, 1, 2],
  "subset": {
    "train": 1024,
    "val": 256,
    "test_iid": 512,
    "test_context": 512,
    "test_combination_101": 512,
    "selection": "fixed_prefix"
  },
  "model": {
    "m_max": 5,
    "token_dim": 64,
    "attention_heads": 4,
    "architecture": "SetMechanismBank"
  },
  "e_step": {
    "num_supports": 32,
    "sample_chunk": 32,
    "support_parallelism": 8,
    "restart_parallelism": 1,
    "v_bound": 1.5,
    "inner_lr": 0.05,
    "solver_ladder": [[20,2], [30,2], [30,3]],
    "lambda_p": 0.001
  },
  "m_step": {
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "batch_size": 128,
    "steps_per_candidate_per_round": 16,
    "gradient_clip": 1.0
  },
  "outer_loop": {
    "max_rounds": 12,
    "min_rounds": 4,
    "stability_window": 3,
    "support_change_threshold": 0.01,
    "val_every_rounds": 2
  },
  "joint_control": {
    "optimizer_updates": 960,
    "batch_size": 128
  }
}
```

---

## 15. 命令模板

```bash
python E0/adp1/run_adp1_mini.py validate --config E0/configs/e0_adp1_mini.json
```

```bash
python E0/adp1/run_adp1_mini.py smoke --config E0/configs/e0_adp1_mini.json --seed 0
```

```bash
python E0/adp1/run_adp1_mini.py validate-solver --config E0/configs/e0_adp1_mini.json --seed 0 --num-samples 16
```

```bash
python E0/adp1/run_adp1_mini.py run-paired --config E0/configs/e0_adp1_mini.json --seed 0
```

```bash
python E0/adp1/run_adp1_mini.py run-paired --config E0/configs/e0_adp1_mini.json --seeds 0,1,2
```

```bash
python E0/adp1/run_adp1_mini.py evaluate --config E0/configs/e0_adp1_mini.json --seeds 0,1,2
```

```bash
python E0/adp1/run_adp1_mini.py summarize --root E0/outputs/e0_adp1_mini
```

可选 amortization：

```bash
python E0/adp1/run_adp1_mini.py amortize --seed 0 --tp identity --max-epochs 30
```

```bash
python E0/adp1/run_adp1_mini.py amortize --seed 0 --tp orthogonal --max-epochs 30
```

---

## 16. Reproducibility checklist

- [ ] 复用原 world/data，未重生成。
- [ ] world、dataset、optimization seeds 分离。
- [ ] 子集是固定前缀，三个 seeds 完全相同。
- [ ] train 中无 `101`，oracle `<1e-10`。
- [ ] `M_max=5`、原 mechanism bank、\(\lambda_P=10^{-3}\)。
- [ ] 不使用 O1/O2/O5/GT checkpoint initialization。
- [ ] ADP1 与 joint 从同一 `initial_full_model.pt` 分叉。
- [ ] E-step 冻结 bank，M-step 冻结 \(m,v\)。
- [ ] 每个 sample 全部枚举 32 supports。
- [ ] support 和 restart 均串流，未同时展开到 GPU。
- [ ] solver A/B/C 只按 reference agreement 选择，三 seeds 固定。
- [ ] peak VRAM 已记录且 `<6GB`；否则先降 support parallelism，再降 sample chunk。
- [ ] 其他 adapter rows 与 optimizer moments 在 candidate update 中不变。
- [ ] 无 candidate pruning、death、reinit、anchor、soft EM 或额外 loss。
- [ ] hidden GT 只由独立 evaluator 读取。
- [ ] checkpoint 只按 val penalized objective 选择。
- [ ] 3 个 seeds 全部报告，未只挑 best seed。
- [ ] paired joint mini control 使用相同 960 updates。
- [ ] NRMSE、F1、instance R²、functional R²、redundancy、101 全部报告。
- [ ] 常数-label 指标标记 N/A。
- [ ] PASS/FAIL 使用预注册标准。

---

## 17. 最终报告必须回答的三句话

1. **在 3 个随机种子中，有多少个从随机初始化进入了 GT-like basin？**
2. **与 byte-paired joint mini baseline 相比，alternating 是否同时改善了结构恢复和冗余关闭，而不只是 reconstruction？**
3. **该结果是 ADP1-MINI PASS、PARTIAL、FAIL，还是因为 mini joint control 也成功而不具诊断性？**

推荐最终表述：

```text
E0-ADP1-MINI：3 个 paired seeds 中 __ 个满足完整结构恢复。
Joint mini baseline 为 __/3，ADP1 为 __/3。
因此，本实验 [支持 / 部分支持 / 不支持 / 无法隔离]“切断 assignment–mechanism
同步 co-adaptation 足以使随机初始化进入 O5 已证明存在的 GT-like basin”。
峰值显存 __ GB，单 seed wall time __。
```

如果 MINI PASS，再决定是否运行 10 seeds 或更大数据；在此之前，不应直接启动原 100k 全规模方案。
