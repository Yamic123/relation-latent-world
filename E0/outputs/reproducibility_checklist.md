# E0 reproducibility checklist

- [x] ground-truth world 参数固定并保存
- [x] world seed / dataset seed / optimization seed 分离
- [x] learner training 未读取 hidden GT
- [x] oracle realizability certificate PASS
- [x] held-out 101 未泄漏至 training
- [x] identity 与 orthogonal TP 均运行 10 seeds
- [x] structure metrics 使用 Hungarian permutation alignment
- [x] realization comparison 使用 affine gauge calibration
- [x] 报告 prediction、participation、functional recovery 和 redundancy
- [x] NC1、NC2、NC4、NC5、NC7 已执行
- [ ] NC3、NC6：E0-A prerequisite 失败，relation stage 按协议未运行
- [ ] relation EP / Figure D 数值矩阵：E0-A prerequisite 失败，标为 N/A
- [x] 每个 run 保存 config、environment、curves、best/final checkpoint、metrics、alignment、decision
- [x] PASS/FAIL 使用预先设定阈值