"""Paired comparison between the original E0-A joint baseline and E0-ADP1."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adp1._utils import ADP1_OUTPUT_ROOT, dump_json


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _joint_metrics(joint_root: Path, tp: str, seed: int) -> Optional[Dict[str, Any]]:
    path = joint_root / tp / f"seed_{seed}" / "metrics.json"
    if not path.exists():
        return None
    m = _read(path)
    usage = np.asarray(m.get("candidate_usage", [0.0] * 5), dtype=np.float64)
    return {
        "method": f"joint {tp}",
        "seed": seed,
        "stable": bool(m.get("pass", False)),
        "iid_nrmse": m["prediction_nrmse"]["test_iid"],
        "context_nrmse": m["prediction_nrmse"].get("test_context", float("nan")),
        "nrmse_101": m["prediction_nrmse"]["test_combination_101"],
        "f1": m.get("participation_f1_mean", float("nan")),
        "inst_r2_min": m.get("instance_effect_r2_min", float("nan")),
        "func_r2_min": m.get("functional_r2_min", float("nan")),
        "redundant_usage": m.get("redundant_usage_max", float("nan")),
        "active_count": float(usage.sum()),
        "pass": bool(m.get("pass", False)),
    }


def _adp1_exact(seed: int) -> Optional[Dict[str, Any]]:
    path = ADP1_OUTPUT_ROOT / f"seed_{seed}" / "discovery" / "metrics_exact.json"
    if not path.exists():
        return None
    m = _read(path)
    return {
        "method": "ADP1 exact",
        "seed": seed,
        "stable": bool(m["pass_components"]["discovery_stable"]),
        "iid_nrmse": m["prediction_nrmse"]["test_iid"],
        "context_nrmse": m["prediction_nrmse"]["test_context"],
        "nrmse_101": m["prediction_nrmse"]["test_combination_101"],
        "f1": m["test_iid"]["participation_f1_mean"],
        "inst_r2_min": m["instance_effect_r2_min"],
        "func_r2_min": m["functional_r2_min"],
        "redundant_usage": m["redundant_usage_max"],
        "active_count": m["test_iid"]["expected_active"],
        "pass": bool(m["pass"]),
    }


def _adp1_amortized(seed: int, tp: str) -> Optional[Dict[str, Any]]:
    path = ADP1_OUTPUT_ROOT / f"seed_{seed}" / "amortization" / tp / "metrics.json"
    if not path.exists():
        return None
    m = _read(path)
    return {
        "method": f"ADP1 q {tp}",
        "seed": seed,
        "stable": True,
        "iid_nrmse": m["prediction_nrmse"]["test_iid"],
        "context_nrmse": float("nan"),
        "nrmse_101": m["prediction_nrmse"]["test_combination_101"],
        "f1": m["participation_f1_mean"],
        "inst_r2_min": m["instance_effect_r2_min"],
        "func_r2_min": float("nan"),
        "redundant_usage": float("nan"),
        "active_count": float("nan"),
        "pass": bool(m["pass"]),
    }


def _bootstrap_ci(diffs: np.ndarray, n: int = 10000) -> Dict[str, float]:
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        means.append(float(np.mean(diffs[idx])))
    means = np.sort(means)
    return {
        "mean": float(np.mean(diffs)),
        "ci_low": float(means[int(0.025 * n)]),
        "ci_high": float(means[int(0.975 * n)]),
        "n": int(len(diffs)),
    }


def compare_joint(joint_root: Path, adp1_root: Path, seeds: List[int] = list(range(10))) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for tp in ["identity", "orthogonal"]:
            j = _joint_metrics(joint_root, tp, seed)
            if j:
                rows.append(j)
        e = _adp1_exact(seed)
        if e:
            rows.append(e)
        for tp in ["identity", "orthogonal"]:
            a = _adp1_amortized(seed, tp)
            if a:
                rows.append(a)

    header = ["seed", "method", "stable", "IID NRMSE", "context NRMSE", "101 NRMSE",
              "F1", "inst-R2 min", "func-R2 min", "redundant usage", "active count", "PASS"]
    csv_path = adp1_root / "paired_baseline_comparison.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in sorted(rows, key=lambda r: (r["seed"], r["method"])):
            writer.writerow([
                r["seed"], r["method"], r["stable"],
                f"{r['iid_nrmse']:.5f}", f"{r['context_nrmse']:.5f}", f"{r['nrmse_101']:.5f}",
                f"{r['f1']:.5f}", f"{r['inst_r2_min']:.5f}", f"{r['func_r2_min']:.5f}",
                f"{r['redundant_usage']:.5f}", f"{r['active_count']:.5f}", r["pass"],
            ])

    # Paired ADP1-exact vs joint (identity) differences, for the shared metrics.
    exact = {s: _adp1_exact(s) for s in seeds}
    joint_id = {s: _joint_metrics(joint_root, "identity", s) for s in seeds}
    paired = {}
    for key, label in [
        ("iid_nrmse", "IID NRMSE"),
        ("nrmse_101", "101 NRMSE"),
        ("f1", "participation F1"),
        ("func_r2_min", "functional R2 min"),
        ("redundant_usage", "redundant usage"),
    ]:
        diffs = []
        for s in seeds:
            if exact[s] and joint_id[s]:
                diffs.append(exact[s][key] - joint_id[s][key])
        if diffs:
            paired[label] = _bootstrap_ci(np.asarray(diffs, dtype=np.float64))

    n_exact_pass = sum(1 for s in seeds if exact[s] and exact[s]["pass"])
    n_stable = sum(1 for s in seeds if exact[s] and exact[s]["stable"])
    summary = {
        "paired_differences_adp1_vs_joint_identity": paired,
        "adp1_exact_pass_seeds": n_exact_pass,
        "adp1_stable_seeds": n_stable,
        "n_seeds": len(seeds),
        "csv": str(csv_path),
    }
    dump_json(adp1_root / "paired_comparison_summary.json", summary)
    return summary
