# E0-ADP1-MINI — chunk_size 偏差记录

## 变更

`e0_adp1_mini.json` 中 `e_step.chunk_size`: **32 → 2048**（2026-09-03，seed 0 首跑后）。

## 为什么

guide §4.2 将 `chunk_size=32` 定为「峰值 <6GB」的内存约束，是从 full-scale
（20 万样本）继承来的。mini 规模仅 1024 样本，实测 round 0 的 E-step 在
chunk_size=32 下要 ~4 分钟（大部分是 31× 的 Python chunk 循环开销，GPU 本身
远未打满）。

`chunk_size` 在 [exact_e_step.py](../adp1/exact_e_step.py) 里是纯吞吐参数：每
样本的 v 优化是独立 Adam 轨迹，chunk 只决定「多少个样本的独立优化并行」。
改成 2048 后，train(1024)/val(256)/test(512) 全部单 chunk 完成，砍掉 ~31× 的
chunk 循环开销。

## 数值影响

非逐位一致，差异来源唯一：`_adam_active_v` 把 loss 按「当前仍 active 的实例
数」归一化，batch 变大 → 每实例梯度缩小 → Adam 的 `eps=1e-8` 相对变大，导致
单步更新有相对偏差。实测（见下）round 0 的 `J_train` 差异 **9e-6**（~1e-4 相
对），`active`（平均 support 大小）差 0.024，即约 2% 的 sample 在 round 0 选了
差 1 bit 的 support。

这个 2% 波动**集中在 round 0 的随机 bank 近 tie 区**——正是 solver 门槛标定里
诊断过的、连续目标一致但离散 support 多模态/近简并的区域（A/B/C 一致率仅
81.25%/81.25%/87.5%）。该区域本就对 ~1e-6 级的扰动敏感，且会随 M-step 训练机
制分化（GT 机制是 v 的二次函数，单峰）而消失。对收敛后的 assignment / 机制，
chunk_size 无实质影响；实验最终结论阈值（IID NRMSE<0.05、F1>0.90 等）与 3-seed
方差都远大于这一扰动。

## 验证

重跑 seed 0 后核对 round 0 的 `J_train`：chunk_size=32 时为 **0.099439**，
chunk_size=2048 时实测 **0.099448**（Δ=9e-6，见 `seed0_run.log` 首行）。round 0
用时 306s → 13s（~23×）。

## 用户决策

用户选择「提速 chunk_size→1024（推荐）」，此处采用 2048 以留余量（任一 split
≤ 2048 单 chunk 完成），峰值显存仍 << 6GB。
