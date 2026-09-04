# Adp3 实验执行方案：先形成 Z_D 身份，再回到 TP 推断参与关系

> **实验定位**
>
> Adp3 用来验证一种新的 Z_D 学习顺序：
>
> \[
> oxed{
> 	ext{先从有对照的 TP 中提取“单一作用片段”}
> ightarrow
> 	ext{把同类片段聚成稳定机制}
> ightarrow
> 	ext{再回到完整 TP 推断 }m,v
> }
> \]
>
> 它不再要求 \(M_{\max}\) 个候选机制从随机初始化开始直接共同解释完整 \(\Delta S\)，因为 E0、Adp1、Adp2 已经表明这很容易形成错误的共同分工。
>
> Adp3 第一阶段继续使用 **E0-A：\(R_D=0\)** 的世界。因此本实验只研究 \(Z_D\) 的形成，不研究机制间关系 \(R_D\)。

---

## 0. 当前已知结论

已有实验已经得到：

1. 给真实 \(m,v\) 时，当前 mechanism network 可以恢复真实机制；
2. 给监督时，当前 inference network 可以恢复 \(m,v\)；
3. 当真实 mechanism basis 已知时，真实 support 基本可辨；
4. 从随机初始化直接学习完整 transformation 时，\(M_{\max}=5\) 个 candidate 会形成错误的共同分工；
5. 精确枚举 support + 交替训练仍不能自动形成正确 mechanism identity；
6. 每个 TP 局部稀疏并不能保证整个世界只保留正确的机制；
7. “第一名解释比第二名好很多”不能作为 mechanism identity 的可靠证据；
8. 仅仅“跨状态区域可复用”也不足以保证 candidate 是真实机制；
9. 当前 E0-A 的数据生成严格满足：
   \[
   \Delta S_i=\sum_{j=1}^{3}e_i^j,\qquad
   e_i^j=m_i^j\mathcal M_j^*(S_i,v_i^j)
   \]
   且没有 \(R_D\) 项。

因此 Adp3 不再从“完整 TP 如何分给 \(M_{\max}\) 个 candidate”开始，而先研究：

\[
oxed{
	ext{能否从 transformation 对照中直接得到单一机制的作用片段，并把这些片段整理成稳定的 }Z_D。
}
\]

---

## 1. Adp3 要验证的核心假设

### 假设 1：完整 transformation 太容易被任意拆分

对于：

\[
\Delta S=e_1+e_2+e_3
\]

只要求：

\[
\hat{\Delta S}=\sum_j\hat e_j
\]

会存在大量不同分解。

因此 reconstruction 适合检查机制集合是否能解释世界，但不适合单独定义机制身份。

### 假设 2：在 \(R_D=0\) 时，有控制的 transformation 对照可以产生更“纯”的机制证据

如果两个 interaction：

- 起始状态相同；
- 其它机制条件相同；
- 只有一个高层作用发生变化；

那么：

\[
d_{ab}
=
\Delta S_b-\Delta S_a
\]

只包含这个变化对应的作用。

### 假设 3：机制身份应该先从许多“单一作用片段”中形成

设提取出大量：

\[
f_n=(S_n,d_n).
\]

如果很多不同 fragment 都可以由同一套：

\[
\mathcal M_j(S,v)
\]

解释，那么它们应该被归为同一个 \(Z_D^j\)。

因此：

\[
oxed{
	ext{先形成 mechanism family，再推断完整 TP 中谁参与}
}
\]

可能比：

\[
oxed{
	ext{先猜完整 TP participation，再希望 candidate 自己形成 identity}
}
\]

更合理。

---

## 2. Adp3 的总体执行顺序

严格按照：

```text
Adp3-0   数据对照是否真的产生“单一作用片段”
   ↓
Adp3-A1  在不知道 M_real 的情况下，片段能否自动形成稳定机制簇
   ↓
Adp3-A2  加入全局机制数量选择，解决 M_max > M_real
   ↓
Adp3-B1  用已经形成的 Z_D 回到完整 TP，推断 participation
   ↓
Adp3-B2  允许 Z_D 在完整 TP 上小幅联合调整，检查 identity 是否保持
   ↓
Adp3-C   补测 TP reconstruction loss 的作用
   ↓
Adp3-F   多 seed 正式确认
```

任何一步 FAIL，先停止并解释，不直接进入后续阶段。

---

## 3. 保持不变的 E0 条件

除本文明确新增的“成组对照采样”外，其余 synthetic world 不变。

```text
WORLD_SEED = 20260901
DATASET_SEED = 20260902
M_real = 3             # evaluator only
M_max = 5              # learner knows only this
TP = identity
state slots = 3
slot dim = 2
v dim = 1
lambda_participation = 1e-3
relation = none
```

真实机制仍为原 E0 generator。

禁止修改：

```text
A_jk
b_jk
c_jk
state distribution
v active support
mechanism architecture
test split definition
heldout 101 definition
```

---

## 4. Adp3 新增的数据：成组对照样本

原 E0 随机训练样本彼此独立，不能保证两条 interaction 具有相同的 \(S,v\) 但只差一个 mechanism。

Adp3 新生成一个 **contrast dataset**。

注意：

\[
oxed{
	ext{改变的是数据采样方式，不改变 synthetic world。}
}
\]

---

## 5. Contrast group 的生成

对每个 base group \(g\)：

先采样一次：

\[
S_g
\]

以及三个 realization：

\[
v_{g1},v_{g2},v_{g3}.
\]

然后固定：

\[
(S_g,v_{g1},v_{g2},v_{g3})
\]

生成所有 8 个 participation pattern：

\[
000,\ 001,\ 010,\ 011,\ 100,\ 101,\ 110,\ 111.
\]

每条样本仍按原 E0 world generator 生成：

\[
\Delta S_{g,m}
=
\sum_jm_j e_{gj}.
\]

---

## 6. 训练时允许 learner 知道什么

learner **不能看到**：

```text
GT mechanism index
GT m vector
GT v_j
GT e_j
```

learner 只接收：

```text
group_id
S
p
DeltaS
pair relation: “这两条 interaction 只相差一个作用因素”
```

第一版实验允许 learner 知道：

\[
oxed{
	ext{pair }(a,b)	ext{ 是 single-change contrast}
}
\]

但**不告诉它改变的是 GT Z1/Z2/Z3 中哪一个**。

因此 pair 本身只提供：

> “这两个世界变化之间只多/少了一种作用”

而不提供：

> “这就是 Z1”。

这是 Adp3 的核心受控信息。

---

## 7. Adp3-0：先验证 contrast fragment 是否干净

### 7.1 提取 fragment

对每一个 single-change pair：

\[
(m_a,m_b)
\]

其中二者 Hamming distance 为 1。

定义：

\[
oxed{
d_{ab}
=
\Delta S_b-\Delta S_a
}
\]

同时固定方向为：

```text
0 → 1
```

即只保留“某个作用从 absent 变为 present”的差分。

每个 base group 中共有：

\[
3	imes2^{3-1}=12
\]

个有向 single-change contrast。

### 7.2 evaluator 检查

虽然 learner 不知道 GT，但 evaluator 知道该 contrast 实际切换的是哪个真实 mechanism。

验证：

\[
d_{ab}
pprox e_j^*(S,v_j).
\]

计算：

```text
fragment MSE
fragment normalized error
```

### Adp3-0 PASS

```text
max fragment MSE < 1e-10
```

如果 FAIL：

STOP。

说明 contrast generator 有 bug，不能继续。

---

## 8. Contrast fragment dataset

最终得到：

\[
\mathcal D_F
=
\{(S_n,d_n)\}_{n=1}^{N_F}.
\]

GT mechanism label：

\[
y_n^*\in\{1,2,3\}
\]

只保存在：

```text
hidden_gt
```

中用于评价。

learner 不读取。

---

## 9. 数据规模

### 开发阶段

```text
base groups train = 1,024
base groups val   = 256
base groups test  = 512
```

每个 base 12 个 contrast：

```text
train fragments = 12,288
val fragments   = 3,072
test fragments  = 6,144
```

如果计算量过大，可先每个 base 随机采样 6 个 contrast，但必须固定采样 seed。

---

## 10. Adp3-A1：先验证“纯 fragment”是否足以形成 mechanism identity

这一阶段暂时**不回到完整 TP**。

只学习：

\[
(S_n,d_n)
\]

中的 mechanism family。

---

## 11. A1 的表示

维护：

\[
M_{\max}=5
\]

个 mechanism candidate：

\[
\mathcal M_1,\dots,\mathcal M_5.
\]

每个 fragment 只能属于一个 candidate：

\[
c_n\in\{1,\dots,5\}.
\]

与以前完整 TP 不同，这里不再使用 multi-hot participation 去表示多个机制共同作用。

因为按照 Adp3-0 的构造，每个 fragment 本身就只包含一个单一作用。

---

## 12. Candidate 对 fragment 的解释误差

对于 fragment \(n\) 与 candidate \(j\)，定义：

\[
E_{nj}
=
\min_v
D_S
\left(
d_n,
\mathcal M_j(S_n,v)
ight).
\]

其中：

\[
v=1.5	anh z.
\]

第一版：

```text
inner optimizer = Adam
inner lr = 0.05
inner steps = 20
restarts = 2
```

保存：

\[
v_{nj}^*
\]

与：

\[
E_{nj}.
\]

---

## 13. A1 使用“分配—拟合”交替训练

这里仍然可以使用类似 EM 的交替过程，但它不再承担“从完整 transformation 中同时创造多个机制”。

它只处理：

\[
oxed{
	ext{一个已经是单一作用的 fragment 应该归到哪个 mechanism family。}
}
\]

### 分配步骤

冻结：

\[
\mathcal M_j.
\]

每个 fragment 选择：

\[
c_n
=
rg\min_j E_{nj}.
\]

### 拟合步骤

冻结：

\[
c_n,v_n.
\]

每个 candidate 只使用自己的 fragments：

\[
\mathcal D_j
=
\{n:c_n=j\}
\]

训练：

\[
\mathcal M_j(S_n,v_n)pprox d_n.
\]

---

## 14. A1 不加入全局数量成本

A1 的目的只回答：

\[
oxed{
	ext{当输入不再是混合 transformation，而是单一作用 fragment 时，}
	ext{random candidate 是否开始形成正确 mechanism identity？}
}
\]

因此：

```text
lambda_global = 0
merge = off
split = off
```

---

## 15. A1 必须记录的逐轮指标

```text
round
train fragment MSE
val fragment NRMSE

candidate usage[5]
candidate heldout error[5]

assignment change rate
exact assignment agreement

functional R2 matrix [5 x 3]
matched functional R2

fragment classification confusion matrix
fragment classification accuracy after Hungarian matching

candidate effect norm
candidate sample count
```

GT matching 只用于 evaluator。

---

## 16. A1 的关键判定

### 强正向信号

如果出现：

```text
3 个 candidate 的 functional R² 明显升高
2 个 candidate 形成重复/低 usage
fragment matching accuracy > 0.90
```

说明：

\[
oxed{
	ext{把“机制身份形成”提前到 fragment 层是正确方向。}
}
\]

### A1 阶段性 PASS

development seed 0：

```text
fragment matching accuracy > 0.90
至少 3 个 candidate matched functional R² > 0.80
min of best-3 functional R² > 0.70
assignment change rate 最后 3 轮 < 0.10
```

此时不要求 alive candidate 恰好等于 3，因为 A1 尚未解决 \(M_{\max}>M_{\mathrm{real}}\)。

---

## 17. 如果 A1 出现 5 个 candidate 分摊三个真实 mechanism

例如：

```text
C1,C4 → 都像 GT Z1
C2    → 像 GT Z2
C3,C5 → 都像 GT Z3
```

这说明：

\[
oxed{
	ext{问题 A：identity family 已形成}
}
\]

但：

\[
oxed{
	ext{问题 B：重复 candidate 还没合并}
}
\]

此时进入 A2。

---

## 18. Adp3-A2：解决 \(M_{\max}>M_{\mathrm{real}}\)

A2 目标：

\[
oxed{
	ext{让功能重复的 candidate 合并，而不是仅靠 local usage 等待其死亡。}
}
\]

---

## 19. 全局存在状态

维护：

\[
g_j\in\{0,1\}.
\]

```text
1 = alive
0 = dormant
```

初始：

\[
g_j=1.
\]

不物理删除 dormant candidate。

---

## 20. Pairwise merge test

对于两个 alive candidates：

\[
j,k
\]

取其 fragment：

\[
\mathcal D_j\cup\mathcal D_k.
\]

训练一个临时共同 mechanism：

\[
\mathcal M_{jk}.
\]

训练只用 train fragments。

然后在 val fragments 比较：

### 分开

\[
L_{\mathrm{sep}}^{val}
\]

### 合并

\[
L_{\mathrm{merge}}^{val}.
\]

定义模型成本：

\[
\lambda_G
\]

表示“长期多保留一个 mechanism”的代价。

接受 merge 当且仅当：

\[
oxed{
L_{\mathrm{merge}}^{val}
+
\lambda_G(K-1)
<
L_{\mathrm{sep}}^{val}
+
\lambda_GK
}
\]

即：

\[
oxed{
L_{\mathrm{merge}}^{val}
-
L_{\mathrm{sep}}^{val}
<
\lambda_G.
}
\]

---

## 21. Merge 必须跨状态验证

为了防止两个 candidate 只是在某一小块状态空间内看起来相似：

将 val fragments 按 \(S\) 分为四个固定 context group。

只有至少：

```text
3 / 4 groups
```

都支持 merge，才接受。

---

## 22. \(\lambda_G\) 的选择

development seed 0：

```text
lambda_G ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
```

不能根据“最终是不是 3 个”选择。

使用：

```text
在 validation fragment NRMSE 距离最优值不超过 2% 的设置中，
选择 alive candidate 最少者。
```

随后固定。

---

## 23. A2 PASS

要求：

```text
alive candidate count = 3
fragment matching accuracy > 0.95
三个 matched functional R² > 0.90
false merge = 0
```

如果 alive=3 但 functional recovery 差：

\[
oxed{
	ext{数量正确不算成功。}
}
\]

---

## 24. A2 的重要负对照

### NC1：只加全局数量成本，不做 fragment 学习

在原 Adp1 full-transformation pipeline 上直接加入同样的 \(\lambda_G\)。

目的：

\[
oxed{
	ext{验证简单让模型少养几个 candidate，并不能自动获得正确 identity。}
}
\]

### NC2：fragment 学习但不 merge

对应 A1。

比较：

```text
A1 vs A2
```

明确区分：

```text
identity formation
```

与：

```text
model count selection
```

---

## 25. Adp3-B1：用形成的 \(Z_D\) 回到完整 TP

只有：

```text
A2 PASS
```

后执行。

此时得到：

\[
\mathcal M_1,\mathcal M_2,\mathcal M_3
\]

但 learner 仍不知道它们对应 GT 哪个编号。

---

## 26. B1 第一阶段冻结 \(Z_D\)

冻结：

\[
\mathcal M_j.
\]

对原始 E0 full-transformation dataset：

\[
(S_i,p_i,\Delta S_i)
\]

重新推断：

\[
m_i,v_i.
\]

因为当前 E0：

\[
M=3
\]

且 \(R_D=0\)，可以直接枚举：

\[
2^3=8
\]

个 support。

求：

\[
(m_i,v_i)
=
rg\min
\left[
D_S
\left(
\Delta S_i,
\sum_jm_i^j\mathcal M_j(S_i,v_i^j)
ight)
+
\lambda_P\|m_i\|_0
ight].
\]

---

## 27. B1 要回答的问题

\[
oxed{
	ext{如果 mechanism identity 已经先形成，
完整 TP 的 participation 是否会变成容易的问题？}
}
\]

这和 O3-E 的区别是：

O3-E 使用 GT mechanism。

B1 使用：

\[
oxed{
	ext{Adp3 自己从 contrast fragments 中学出的 mechanism。}
}
\]

---

## 28. B1 PASS

原 E0 阈值：

```text
IID NRMSE < 0.05
101 NRMSE < 0.10
mean participation F1 > 0.90
min instance-effect R² > 0.90
min functional R² > 0.90
```

101 中：

```text
Z1/Z3 effect R² > 0.90
inactive Z2 FPR < 0.10
```

---

## 29. Adp3-B2：完整 TP 上的小幅联合调整

B1 PASS 后才允许。

目标：

> 验证已经形成的 mechanism identity 在 full TP reconstruction 下是否会保持，而不是重新掉回 distributed basin。

---

## 30. B2 loss

第一版只使用：

\[
oxed{
L_{\mathrm{effect}}
+
\lambda_P L_{\mathrm{participation}}
}
\]

暂时仍令：

\[
\lambda_T=0.
\]

原因：

先与原 E0 objective 保持完全可比。

---

## 31. B2 更新原则

```text
mechanism learning rate = 原随机训练的 0.1 倍
inference variables / assignments 正常更新
```

每轮监控：

```text
functional R²
candidate identity mapping
participation F1
IID NRMSE
101 NRMSE
```

### B2 PASS

训练后：

```text
min functional R² 下降 < 0.05
participation F1 不下降
prediction 维持或改善
```

---

## 32. Adp3-C：补测 TP reconstruction loss

这是对完整数学建模的桥接实验。

只有 B2 稳定后运行。

比较：

### C0

\[
L_{\mathrm{effect}}
+
\lambda_PL_{\mathrm{participation}}
\]

### C1

\[
L_{\mathrm{effect}}
+
\lambda_PL_{\mathrm{participation}}
+
\lambda_TL_{\mathrm{TP-rec}}.
\]

---

## 33. TP reconstruction

继续原设计：

\[
\hat p_i
=
G_P(z_i^D)
\]

\[
L_{\mathrm{TP-rec}}
=
D_P(p_i,\hat p_i).
\]

其中 E0：

\[
p_i=B\,\operatorname{vec}(\Delta S_i).
\]

---

## 34. C 阶段要回答的问题

不是：

> TP-rec 能不能让 prediction 更低。

而是：

\[
oxed{
	ext{TP-rec 是否帮助保持 transformation information，
并且不会破坏已经形成的 mechanism identity？}
}
\]

报告：

```text
TP reconstruction error
IID NRMSE
101 NRMSE
participation F1
functional R²
```

---

## 35. 关键诊断：fragment 是否真的比 full TP 更容易学

最终必须有直接对照：

| 方法 | 输入学习对象 | Mmax | identity recovery | count recovery | full TP recovery |
|---|---|---:|---:|---:|---:|
| 原 E0 joint | 完整 \(\Delta S\) | 5 | FAIL | FAIL | FAIL |
| Adp1 | 完整 \(\Delta S\) | 5 | FAIL | FAIL | FAIL |
| Adp3-A1 | 单一 fragment | 5 | ? | 不要求 | N/A |
| Adp3-A2 | 单一 fragment | 5 | ? | ? | N/A |
| Adp3-B1 | 完整 TP，冻结已学 \(Z_D\) | learned | ? | ? | ? |
| Adp3-B2 | 完整 TP，微调 \(Z_D\) | learned | ? | ? | ? |

---

## 36. Adp3 的失败分支

### 情况 1：Adp3-0 FAIL

contrast 数据生成错误。

STOP。

### 情况 2：A1 仍完全无法形成 GT-like function families

说明：

\[
oxed{
	ext{即使给 learner 单一作用 fragment，
当前机制表示/聚合方式仍不能形成高层 identity。}
}
\]

这时不要进入 full TP。

### 情况 3：A1 能形成三个 function family，但 5 个槽位都活着

说明：

\[
oxed{
	ext{问题 A 基本解决，问题 B 仍在。}
}
\]

进入 A2。

### 情况 4：A2 错误合并不同 GT mechanisms

说明：

\[
oxed{
	ext{当前“一个共同 function 能不能解释两组 fragment”的判据还不够。}
}
\]

### 情况 5：A2 PASS，但 B1 FAIL

说明：

\[
oxed{
	ext{单一 mechanism identity 能形成，
但把它们重新组合回完整 TP 时仍存在 participation / composition 问题。}
}
\]

### 情况 6：B1 PASS，B2 FAIL

说明：

\[
oxed{
	ext{机制能发现，但 full TP joint fine-tuning 会重新破坏 identity。}
}
\]

### 情况 7：B2 PASS，C1 FAIL

说明：

\[
oxed{
L_{\mathrm{TP-rec}}
	ext{ 与当前 mechanism identity 存在冲突。}
}
\]

---

## 37. 多 seed 正式确认

只有 B1/B2 development seed 0 通过后运行。

### 中等规模

```text
optimization seeds = 0,1,2
base contrast groups = 4096
```

至少：

```text
2/3
```

通过再进入正式确认。

### 正式

```text
optimization seeds = 0..9
```

Strong pass：

\[
oxed{
RecoveryRate\ge 8/10.
}
\]

---

## 38. 输出目录

```text
outputs/
└── e0_adp3/
    ├── configs/
    ├── contrast_data/
    │   ├── visible_train.npz
    │   ├── visible_val.npz
    │   ├── visible_test.npz
    │   └── hidden_gt.npz
    │
    ├── adp3_0/
    ├── a1_seed_0/
    │   ├── round_metrics.jsonl
    │   ├── checkpoints/
    │   ├── fragment_assignments/
    │   └── final_metrics.json
    │
    ├── a2_seed_0/
    ├── b1_seed_0/
    ├── b2_seed_0/
    ├── c0_seed_0/
    ├── c1_seed_0/
    └── full/
```

---

## 39. 每轮必须保存

```text
round_metrics.jsonl
```

字段：

```text
round
train_fragment_mse
val_fragment_nrmse
candidate_usage[5]
candidate_sample_count[5]

assignment_change_rate
exact_assignment_agreement

functional_r2_matrix
matched_functional_r2
fragment_matching_accuracy

alive[5]
num_alive

merge_proposals
merge_accepts

e_step_seconds
m_step_seconds
merge_seconds
```

---

## 40. 推荐命令模板

实际脚本名可调整，但 stage 名保持一致。

```bash
python run_adp3.py --stage adp3_0 --seed 0
python run_adp3.py --stage adp3_a1 --seed 0
python run_adp3.py --stage adp3_a2 --seed 0
python run_adp3.py --stage adp3_b1 --seed 0
python run_adp3.py --stage adp3_b2 --seed 0
python run_adp3.py --stage adp3_c --seed 0
```

---

## 41. 负对照

### NC-1：打乱 contrast pair

随机打乱 pair 中第二条 interaction。

此时：

\[
d_{ab}
\]

不再对应单一作用。

预期：

```text
fragment clustering 失败
functional recovery 明显下降
```

### NC-2：直接用完整 \(\Delta S\) 做单一归类

把 fragment 替换为：

\[
d_n=\Delta S_n.
\]

仍使用 A1 相同聚合算法。

预期不能稳定形成三个 GT mechanism families。

### NC-3：A1 成功后随机打乱 fragment assignments 再拟合

检查 functional recovery 是否消失。

---

## 42. 需要特别警惕的“假成功”

以下都不能单独算成功：

```text
fragment reconstruction 很低
candidate 数量变成 3
某三个 candidate usage 很高
full TP NRMSE 下降
```

真正成功必须满足：

\[
oxed{
	ext{fragment identity 正确}
+
	ext{机制数量正确}
+
	ext{full TP participation 正确}
+
	ext{101 组合泛化正确}
}
\]

---

## 43. Adp3 最终科学问题

Adp3 最终不是为了证明“聚类比 EM 好”。

它要验证的是：

\[
oxed{
	extbf{一个高层 }Z_D	extbf{ 是否应该先从跨 TP 的稳定 transformation fragments 中形成身份，}
}
\]

然后再作为已经存在的高层机制去解释完整 TP。

如果 Adp3 成功，则支持新的学习顺序：

\[
oxed{
	ext{TP 对照}
ightarrow
	ext{单一作用 fragments}
ightarrow
	ext{mechanism families}
ightarrow
Z_D
ightarrow
	ext{完整 TP participation}
}
\]

而不是原来的：

\[
oxed{
	ext{完整 TP}
ightarrow
	ext{随机 }M_{\max}	ext{ 个 candidate 同时竞争}
ightarrow
	ext{期待 }Z_D	ext{ 自发形成}.
}
\]

如果 Adp3-A1 在 clean fragments 上仍失败，那么下一步就不应继续研究 participation 或全局稀疏，而应直接重新审查：

\[
oxed{
	ext{我们对“高层 mechanism family 如何由低层 transformation evidence 表示和比较”的建模是否正确。}
}
\]

---

## 44. 一句话执行标准

\[
oxed{
	ext{先证明“单一作用片段能自动聚成稳定 }Z_D	ext{”，
再允许这些 }Z_D	ext{ 回去解释完整 TP。}
}
\]

这是 Adp3 相对于 E0、Adp1、Adp2 的唯一核心路线变化。
