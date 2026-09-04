# E0 ADP3-A1 结果摘要

- PASS: **False**
- rounds: 40
- final train fragment MSE: 0.001507
- final val fragment NRMSE: 0.145912
- corrected fragment Hungarian accuracy: 0.567383
- matched functional R²: [0.888623, 0.323209, -0.176002]
- last-3 assignment change rate: 0.319661
- candidate usage: [0.210042, 0.140544, 0.311605, 0.109945, 0.227865]
- CUDA peak allocated: 48.94 MiB

按执行方案，A1 FAIL 后停止；A2、B1、B2、C 和多 seed 未运行。

注意：原始 round_metrics.jsonl 的 fragment_matching_accuracy 使用了 functional mapping。
权威分类指标位于 corrected_round_metrics.jsonl 和 final_verdict.json。
