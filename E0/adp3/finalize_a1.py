"""Merge the corrected evaluator audit into authoritative ADP3-A1 artifacts."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E0_ROOT = Path(__file__).resolve().parent.parent
if str(E0_ROOT) not in sys.path:
    sys.path.insert(0, str(E0_ROOT))
from adp3.run_adp3 import OUT, config, dump_json


def main() -> None:
    c = config()
    out = OUT / "a1_seed_0"
    raw = [json.loads(x) for x in (out / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    audit = json.loads((out / "corrected_evaluator_metrics.json").read_text(encoding="utf-8"))["rounds"]
    assert len(raw) == len(audit) == c["a1"]["rounds"]
    merged = []
    for row, fixed in zip(raw, audit):
        assert row["round"] == fixed["round"]
        corrected = dict(row)
        corrected.update(fixed)
        corrected["metric_correction_note"] = "fragment_matching_accuracy uses assignment-contingency Hungarian matching"
        merged.append(corrected)
    with (out / "corrected_round_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "round_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["round", "train_mse", "val_nrmse", "cluster_accuracy", "assignment_change",
                    "functional_r2_1", "functional_r2_2", "functional_r2_3", "u1", "u2", "u3", "u4", "u5"])
        for x in merged:
            w.writerow([x["round"], x["train_fragment_mse"], x["val_fragment_nrmse"],
                        x["fragment_matching_accuracy"], x["assignment_change_rate"],
                        *x["matched_functional_r2"], *x["candidate_usage"]])
    last = merged[-1]
    last3_change = float(np.mean([x["assignment_change_rate"] for x in merged[-3:]]))
    th = c["pass_thresholds"]
    checks = {
        "fragment_matching_accuracy_gt_0_90": bool(last["fragment_matching_accuracy"] > th["fragment_matching_accuracy_strict_gt"]),
        "each_matched_functional_r2_gt_0_80": bool(all(x > th["matched_functional_r2_each"] for x in last["matched_functional_r2"])),
        "min_best3_functional_r2_gt_0_70": bool(min(last["matched_functional_r2"]) > th["matched_functional_r2_min"]),
        "last3_assignment_change_lt_0_10": bool(last3_change < th["last3_assignment_change_rate"]),
    }
    verdict = {
        "stage": "adp3_a1",
        "authoritative_corrected_evaluator": True,
        "rounds": len(merged),
        "final_train_fragment_mse": last["train_fragment_mse"],
        "final_val_fragment_nrmse": last["val_fragment_nrmse"],
        "final_fragment_matching_accuracy": last["fragment_matching_accuracy"],
        "final_matched_functional_r2": last["matched_functional_r2"],
        "final_functional_r2_matrix": last["functional_r2_matrix"],
        "final_candidate_usage": last["candidate_usage"],
        "final_candidate_sample_count": last["candidate_sample_count"],
        "final_assignment_contingency": last["assignment_contingency_rows_candidate_cols_gt"],
        "last3_assignment_change_rate": last3_change,
        "restart_assignment_agreement": last["exact_assignment_agreement"],
        "cuda_peak_allocated_mib": last["cuda_peak_allocated_mib"],
        "checks": checks,
        "pass": bool(all(checks.values())),
        "stop_after_a1": not all(checks.values()),
        "a2_b1_b2_c_executed": False,
        "stop_reason": "A1 identity formation gate failed; execution plan requires stopping before A2." if not all(checks.values()) else None,
    }
    dump_json(out / "final_verdict.json", verdict)
    make_figures(out, merged)
    lines = [
        "# E0 ADP3-A1 结果摘要", "",
        f"- PASS: **{verdict['pass']}**",
        f"- rounds: {len(merged)}",
        f"- final train fragment MSE: {last['train_fragment_mse']:.6f}",
        f"- final val fragment NRMSE: {last['val_fragment_nrmse']:.6f}",
        f"- corrected fragment Hungarian accuracy: {last['fragment_matching_accuracy']:.6f}",
        f"- matched functional R²: {[round(x, 6) for x in last['matched_functional_r2']]}",
        f"- last-3 assignment change rate: {last3_change:.6f}",
        f"- candidate usage: {[round(x, 6) for x in last['candidate_usage']]}",
        f"- CUDA peak allocated: {last['cuda_peak_allocated_mib']:.2f} MiB", "",
        "按执行方案，A1 FAIL 后停止；A2、B1、B2、C 和多 seed 未运行。", "",
        "注意：原始 round_metrics.jsonl 的 fragment_matching_accuracy 使用了 functional mapping。",
        "权威分类指标位于 corrected_round_metrics.jsonl 和 final_verdict.json。", "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_figures(out: Path, rows: list[dict]) -> None:
    r = np.asarray([x["round"] for x in rows])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(r, [x["val_fragment_nrmse"] for x in rows]); axes[0, 0].set_title("Validation fragment NRMSE")
    axes[0, 1].plot(r, [x["fragment_matching_accuracy"] for x in rows]); axes[0, 1].axhline(.9, ls="--", c="k"); axes[0, 1].set_title("Assignment Hungarian accuracy")
    fr = np.asarray([x["matched_functional_r2"] for x in rows])
    for j in range(3): axes[1, 0].plot(r, fr[:, j], label=f"match {j+1}")
    axes[1, 0].axhline(.8, ls="--", c="k"); axes[1, 0].set_title("Matched functional R2"); axes[1, 0].legend()
    axes[1, 1].plot(r, [x["assignment_change_rate"] for x in rows]); axes[1, 1].axhline(.1, ls="--", c="k"); axes[1, 1].set_title("Assignment change rate")
    for ax in axes.flat: ax.set_xlabel("round"); ax.grid(alpha=.25)
    fig.savefig(out / "a1_dynamics.png", dpi=160); plt.close(fig)
    usage = np.asarray([x["candidate_usage"] for x in rows])
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for j in range(usage.shape[1]): ax.plot(r, usage[:, j], label=f"C{j+1}")
    ax.set(xlabel="round", ylabel="usage", title="Candidate usage trajectories"); ax.grid(alpha=.25); ax.legend(ncol=5)
    fig.savefig(out / "a1_candidate_usage.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
