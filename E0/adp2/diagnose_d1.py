"""ADP2-D1: read-only identity dynamics diagnostic over saved A1 rounds."""
from __future__ import annotations

import csv, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import run_e0 as base
from adp1._utils import DEVICE, dump_json
from adp2.run_compact import cfg, context_spec, reuse, visible

A1 = ROOT / "outputs" / "e0_adp2_compact" / "a1_seed_0"
OUT = ROOT / "outputs" / "e0_adp2_compact" / "d1_identity_dynamics"
BINS = [(0, .01, "0~0.01"), (.01, .02, "0.01~0.02"), (.02, .05, "0.02~0.05"),
        (.05, .10, "0.05~0.10"), (.10, .20, "0.10~0.20"), (.20, np.inf, ">0.20")]


class Holder:
    def __init__(self, bank): self.bank = bank


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float | None:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.std(x[good]) == 0 or np.std(y[good]) == 0: return None
    return float(fn(x[good], y[good]).statistic)


def bucket_rows(margin: np.ndarray, strict: np.ndarray, matched: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for lo, hi, label in BINS:
        take = (margin >= lo) & (margin < hi)
        rows.append({"bin": label, "count": int(take.sum()),
                     "strict_gt_support_accuracy": float(strict[take].mean()) if take.any() else None,
                     "matched_three_accuracy": float(matched[take].mean()) if take.any() else None,
                     "mean_margin": float(margin[take].mean()) if take.any() else None})
    return rows


def main() -> None:
    c = cfg(); OUT.mkdir(parents=True, exist_ok=True)
    S, p, delta = visible("train", c["subset"]["train"])
    hidden = {k: v[:len(S)] for k, v in base.load_hidden("e0a", "train").items()}
    q, threshold, groups = context_spec(S, c["context_seed"])
    round_rows, margin_detail = [], []
    started = time.time()
    for r in range(12):
        t = time.time()
        assignment = np.load(A1 / "assignments" / f"round_{r:03d}.npz")
        m, v, margin = assignment["m"], assignment["v"], assignment["relative_margin"]
        bank = base.SetMechanismBank(5, 64, 4).to(DEVICE)
        bank.load_state_dict(torch.load(A1 / "checkpoints" / f"round_{r:03d}_bank.pt",
                                        map_location=DEVICE, weights_only=True))
        reuse_rows = reuse(bank, S, delta, m, v, groups, range(5), c)
        func, _, _ = base.functional_matrix(Holder(bank), v, hidden, n_probe=5000)
        hr, hc = linear_sum_assignment(-np.nan_to_num(func, nan=-1e6))
        mapping = {int(gt): int(learned) for learned, gt in zip(hr, hc)}
        learned3 = np.column_stack([m[:, mapping[g]] for g in range(3)])
        matched_correct = np.all(learned3 == hidden["m_gt"], axis=1)
        redundant = [j for j in range(5) if j not in set(mapping.values())]
        strict_correct = matched_correct & np.all(m[:, redundant] == 0, axis=1)
        by_bin = bucket_rows(margin, strict_correct, matched_correct)
        margin_detail.append({"round": r, "mapping_gt_to_learned": mapping, "bins": by_bin})
        reuse_values = [x["reuse_error"] for x in reuse_rows]
        best_gt = np.argmax(func, axis=1)
        best_r2 = np.max(func, axis=1)
        round_rows.append({"round": r, "reuse": reuse_values,
                           "reuse_context_coverage": [x["coverage"] for x in reuse_rows],
                           "functional_r2_matrix": func.tolist(), "best_gt_per_candidate": best_gt.tolist(),
                           "best_functional_r2_per_candidate": best_r2.tolist(),
                           "hungarian_mapping_gt_to_learned": mapping,
                           "strict_gt_support_accuracy": float(strict_correct.mean()),
                           "matched_three_accuracy": float(matched_correct.mean()),
                           "margin_bins": by_bin, "seconds": time.time() - t})
        print(f"[D1] round={r:02d} reuse={np.round(reuse_values,3).tolist()} "
              f"bestR2={np.round(best_r2,3).tolist()} strict={strict_correct.mean():.3f}", flush=True)

    reuse_mat = np.array([x["reuse"] for x in round_rows], dtype=float)
    r2_mat = np.array([x["best_functional_r2_per_candidate"] for x in round_rows], dtype=float)
    correlations = {"pooled_pearson_reuse_vs_best_r2": safe_corr(pearsonr, reuse_mat.ravel(), r2_mat.ravel()),
                    "pooled_spearman_reuse_vs_best_r2": safe_corr(spearmanr, reuse_mat.ravel(), r2_mat.ravel()),
                    "per_candidate": []}
    for j in range(5):
        correlations["per_candidate"].append({"candidate": j,
            "pearson": safe_corr(pearsonr, reuse_mat[:, j], r2_mat[:, j]),
            "spearman": safe_corr(spearmanr, reuse_mat[:, j], r2_mat[:, j]),
            "lag1_spearman_reuse_r_to_r2_next": safe_corr(spearmanr, reuse_mat[:-1, j], r2_mat[1:, j])})

    all_margin = []
    all_strict = []
    all_matched = []
    for r, detail in enumerate(margin_detail):
        a = np.load(A1 / "assignments" / f"round_{r:03d}.npz")
        m, margin = a["m"], a["relative_margin"]
        mapping = detail["mapping_gt_to_learned"]
        learned3 = np.column_stack([m[:, mapping[str(g)] if str(g) in mapping else mapping[g]] for g in range(3)])
        matched = np.all(learned3 == hidden["m_gt"], axis=1)
        used = {mapping[str(g)] if str(g) in mapping else mapping[g] for g in range(3)}
        strict = matched & np.all(m[:, [j for j in range(5) if j not in used]] == 0, axis=1)
        all_margin.append(margin); all_strict.append(strict); all_matched.append(matched)
    aggregate_bins = bucket_rows(np.concatenate(all_margin), np.concatenate(all_strict), np.concatenate(all_matched))

    transients = []
    for j in range(5):
        peak = int(np.argmax(r2_mat[:, j]))
        transients.append({"candidate": j, "peak_round": peak, "peak_best_functional_r2": float(r2_mat[peak,j]),
                           "peak_gt": int(round_rows[peak]["best_gt_per_candidate"][j]),
                           "final_best_functional_r2": float(r2_mat[-1,j]),
                           "drop_after_peak": float(r2_mat[peak,j] - r2_mat[-1,j]),
                           "ever_above_0_5": bool(np.any(r2_mat[:,j] > .5)),
                           "ever_above_0_9": bool(np.any(r2_mat[:,j] > .9))})
    final = {"experiment": "ADP2-D1", "read_only_no_a1_retraining": True,
             "checkpoint_assignment_pairing": "round-r post-M bank with saved round-r pre-M assignment; v is reoptimized for reuse",
             "rounds": round_rows, "reuse_matrix_12x5": reuse_mat.tolist(),
             "best_functional_r2_matrix_12x5": r2_mat.tolist(), "correlations": correlations,
             "aggregate_margin_bins": aggregate_bins, "transient_identity": transients,
             "wall_seconds": time.time() - started}
    dump_json(OUT / "final_metrics.json", final)
    with (OUT / "reuse_12x5.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["round","C1","C2","C3","C4","C5"])
        for r in range(12): w.writerow([r, *reuse_mat[r].tolist()])
    with (OUT / "best_functional_r2_12x5.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["round","C1","C2","C3","C4","C5"])
        for r in range(12): w.writerow([r, *r2_mat[r].tolist()])
    write_summary(final)
    print(json.dumps({"correlations": correlations, "aggregate_margin_bins": aggregate_bins,
                      "transient_identity": transients, "wall_seconds": final["wall_seconds"]},
                     ensure_ascii=False, indent=2))


def _write_summary_legacy(x: dict[str, Any]) -> None:
    reuse_mat = np.array(x["reuse_matrix_12x5"]); r2_mat = np.array(x["best_functional_r2_matrix_12x5"])
    lines = ["# ADP2-D1：逐轮机制身份诊断", "", "## Reuse error（12×5）", "",
             "| Round | C1 | C2 | C3 | C4 | C5 |", "|---:|---:|---:|---:|---:|---:|"]
    for r in range(12): lines.append("| " + " | ".join([str(r), *[f"{z:.4f}" for z in reuse_mat[r]]]) + " |")
    lines += ["", "## 每个 candidate 对任一 GT 的最佳 functional R²", "",
              "| Round | C1 | C2 | C3 | C4 | C5 |", "|---:|---:|---:|---:|---:|---:|"]
    for r in range(12): lines.append("| " + " | ".join([str(r), *[f"{z:.4f}" for z in r2_mat[r]]]) + " |")
    lines += ["", "## 相关性", "", f"- Pooled Pearson(reuse, best R²): {x['correlations']['pooled_pearson_reuse_vs_best_r2']}",
              f"- Pooled Spearman(reuse, best R²): {x['correlations']['pooled_spearman_reuse_vs_best_r2']}",
              "- 负相关才支持 reuse error 下降伴随 GT functional recovery 上升。", "", "## Margin 分桶（全部轮次）", "",
              "| Δ bin | N | Strict support accuracy | Matched-three accuracy |", "|---|---:|---:|---:|"]
    for b in x["aggregate_margin_bins"]:
        lines.append(f"| {b['bin']} | {b['count']} | {b['strict_gt_support_accuracy']:.4f} | {b['matched_three_accuracy']:.4f} |")
    lines += ["", "## 短暂接近 GT", ""]
    for t in x["transient_identity"]:
        lines.append(f"- C{t['candidate']+1}: peak round {t['peak_round']}, peak R²={t['peak_best_functional_r2']:.4f}, final={t['final_best_functional_r2']:.4f}, drop={t['drop_after_peak']:.4f}.")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(x: dict[str, Any]) -> None:
    reuse_mat = np.array(x["reuse_matrix_12x5"])
    r2_mat = np.array(x["best_functional_r2_matrix_12x5"])
    lines = [
        "# ADP2-D1：逐轮机制身份诊断", "", "## Reuse error（12×5）", "",
        "| Round | C1 | C2 | C3 | C4 | C5 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in range(12):
        lines.append("| " + " | ".join([str(r), *[f"{z:.4f}" for z in reuse_mat[r]]]) + " |")
    lines += [
        "", "## 每个 candidate 对任一 GT 的最佳 functional R²", "",
        "| Round | C1 | C2 | C3 | C4 | C5 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in range(12):
        lines.append("| " + " | ".join([str(r), *[f"{z:.4f}" for z in r2_mat[r]]]) + " |")
    lines += [
        "", "## 相关性", "",
        f"- Pooled Pearson(reuse, best R²): {x['correlations']['pooled_pearson_reuse_vs_best_r2']}",
        f"- Pooled Spearman(reuse, best R²): {x['correlations']['pooled_spearman_reuse_vs_best_r2']}",
        "- 只有负相关才支持 reuse error 下降伴随 GT functional recovery 上升。",
        "", "## Margin 分桶（全部轮次）", "",
        "| Δ bin | N | Strict support accuracy | Matched-three accuracy |",
        "|---|---:|---:|---:|",
    ]
    for b in x["aggregate_margin_bins"]:
        lines.append(
            f"| {b['bin']} | {b['count']} | {b['strict_gt_support_accuracy']:.4f} | "
            f"{b['matched_three_accuracy']:.4f} |"
        )
    lines += ["", "## 短暂接近 GT", ""]
    for t in x["transient_identity"]:
        lines.append(
            f"- C{t['candidate']+1}: peak round {t['peak_round']}, "
            f"peak R²={t['peak_best_functional_r2']:.4f}, "
            f"final={t['final_best_functional_r2']:.4f}, drop={t['drop_after_peak']:.4f}."
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
