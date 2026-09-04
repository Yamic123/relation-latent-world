"""Diagnose how an E0-A mechanism bank degenerates on a chosen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import run_e0 as e0


def analyze(tp: str, seed: int, split: str) -> Dict[str, Any]:
    S, p, target = e0.load_visible("e0a", split, tp)
    model = e0.load_base_model(tp, seed)
    output = e0.predict_a(model, S, p)

    gates = output["gate_probs"]
    realization = output["v"]
    effects = output["effects"]
    raw_effects = output["raw"]
    prediction = output["prediction"]
    n, m_max = gates.shape

    flat_effects = effects.reshape(n, m_max, -1)
    effect_norms = np.linalg.norm(flat_effects, axis=2)
    raw_norms = np.linalg.norm(raw_effects.reshape(n, m_max, -1), axis=2)
    total_norm = np.linalg.norm(prediction.reshape(n, -1), axis=1)
    full_mse = float(np.mean((target - prediction) ** 2))

    run_dir = e0.run_dir_a(tp, seed)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    learned_to_gt = {
        int(learned): int(gt)
        for gt, learned in metrics["alignment_gt_to_learned"].items()
    }

    rows: List[Dict[str, Any]] = []
    for candidate in range(m_max):
        ablated_prediction = prediction - effects[:, candidate]
        ablated_mse = float(np.mean((target - ablated_prediction) ** 2))
        rows.append(
            {
                "candidate": candidate + 1,
                "hungarian_aligned_gt": learned_to_gt.get(candidate + 1),
                "E_m": float(gates[:, candidate].mean()),
                "std_m": float(gates[:, candidate].std()),
                "E_abs_v": float(np.abs(realization[:, candidate]).mean()),
                "mean_v": float(realization[:, candidate].mean()),
                "std_v": float(realization[:, candidate].std()),
                "E_effect_norm": float(effect_norms[:, candidate].mean()),
                "std_effect_norm": float(effect_norms[:, candidate].std()),
                "E_raw_effect_norm": float(raw_norms[:, candidate].mean()),
                "leave_one_out_EP_mse": ablated_mse - full_mse,
            }
        )

    effect_cosine = np.eye(m_max, dtype=np.float64)
    for first in range(m_max):
        for second in range(first + 1, m_max):
            a = flat_effects[:, first]
            b = flat_effects[:, second]
            denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            sample_cosine = np.sum(a * b, axis=1) / np.maximum(denominator, 1e-12)
            effect_cosine[first, second] = effect_cosine[second, first] = float(
                sample_cosine.mean()
            )

    norm_ratio = effect_norms.sum(axis=1) / np.maximum(total_norm, 1e-12)
    hidden = e0.load_hidden("e0a", split)
    pattern_code = (hidden["m_gt"].astype(np.int64) * np.asarray([4, 2, 1])).sum(axis=1)
    pattern_diagnostics: Dict[str, Any] = {}
    for code in sorted(np.unique(pattern_code)):
        mask = pattern_code == code
        pattern = format(int(code), "03b")
        individual_sum = effect_norms[mask].sum(axis=1)
        group_total_norm = total_norm[mask]
        target_norm = np.linalg.norm(target[mask].reshape(int(mask.sum()), -1), axis=1)
        pattern_diagnostics[pattern] = {
            "n": int(mask.sum()),
            "mean_target_norm": float(target_norm.mean()),
            "mean_prediction_norm": float(group_total_norm.mean()),
            "mean_sum_individual_norms": float(individual_sum.mean()),
            "ratio_of_mean_norms": float(
                individual_sum.mean() / max(group_total_norm.mean(), 1e-12)
            ),
            "cancellation_fraction": float(
                1.0 - group_total_norm.mean() / max(individual_sum.mean(), 1e-12)
            ),
        }
    result: Dict[str, Any] = {
        "tp": tp,
        "optimization_seed": seed,
        "split": split,
        "n": n,
        "checkpoint": "best_checkpoint.pt",
        "gate_mode": "deterministic expected Hard-Concrete nonzero probability",
        "full_mse": full_mse,
        "mean_total_effect_norm": float(total_norm.mean()),
        "rows": rows,
        "mean_sum_individual_norm_over_total_norm": float(norm_ratio.mean()),
        "median_sum_individual_norm_over_total_norm": float(np.median(norm_ratio)),
        "effect_pairwise_mean_sample_cosine": effect_cosine.tolist(),
        "v_correlation": np.corrcoef(realization, rowvar=False).tolist(),
        "pattern_diagnostics": pattern_diagnostics,
    }
    output_path = run_dir / f"degeneracy_diagnostics_{split}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", choices=["identity", "orthogonal"], default="identity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=list(e0.SPLIT_SIZES), default="test_iid")
    args = parser.parse_args()
    print(json.dumps(analyze(args.tp, args.seed, args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
