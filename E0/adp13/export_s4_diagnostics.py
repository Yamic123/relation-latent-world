"""Package selected S4 round diagnostics without rerunning training."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

_DLL_HANDLE = None
_DLL_DIR = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and _DLL_DIR.is_dir():
    os.environ["PATH"] = str(_DLL_DIR) + os.pathsep + os.environ.get("PATH", "")
    _DLL_HANDLE = os.add_dll_directory(str(_DLL_DIR))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import adp6.run_adp6 as adp6

ROUNDS = np.asarray([0, 1, 2, 5, 10, 15, 20, 30, 39], dtype=np.int16)
SOURCE = ROOT / "outputs" / "e0_adp13_sparse_bayesian" / "s4_top1_evidence"
TARGET = ROOT / "outputs" / "e0_adp13_sparse_bayesian" / "diagnostic_exports"


def load_metrics(path: Path):
    by_round = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_round[int(row["round"])] = row
    return by_round


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export_seed(seed: int, family_gt: np.ndarray):
    root = SOURCE / f"seed_{seed:03d}"
    metrics = load_metrics(root / "training" / "round_metrics.jsonl")
    payload = {
        "rounds": ROUNDS,
        "family_gt_id": family_gt.astype(np.int8, copy=False),
    }
    audit = {}
    for r in ROUNDS.tolist():
        saved = np.load(root / "training" / "assignments" / f"round_{r:03d}.npz")
        assignments = saved["train"].astype(np.int8, copy=False)
        costs = saved["train_cost"].astype(np.float32, copy=False)
        row = metrics[r]
        alive = np.asarray(row.get("alive_after_selection", row["alive"]), dtype=np.int8)
        payload[f"assignments_r{r}"] = assignments
        payload[f"costs_r{r}"] = costs
        payload[f"candidate_a_r{r}"] = np.asarray(row["candidate_a"], dtype=np.float32)
        payload[f"candidate_b_r{r}"] = np.asarray(row["candidate_b"], dtype=np.float32)
        payload[f"alive_candidates_r{r}"] = alive
        if assignments.shape != family_gt.shape:
            raise ValueError(f"seed {seed} round {r}: assignment/GT shape mismatch")
        if costs.shape != (len(family_gt), 5) or not np.isfinite(costs).all():
            raise ValueError(f"seed {seed} round {r}: invalid cost matrix")
        if not set(np.unique(assignments)).issubset(set(alive.tolist())):
            raise ValueError(f"seed {seed} round {r}: assignment contains a deleted candidate")
        audit[str(r)] = {
            "assignment_shape": list(assignments.shape),
            "cost_shape": list(costs.shape),
            "alive_candidates_zero_based": alive.tolist(),
            "assignment_histogram_C1_to_C5": np.bincount(assignments, minlength=5).tolist(),
        }
    TARGET.mkdir(parents=True, exist_ok=True)
    out = TARGET / f"s4_seed{seed}_diagnostics.npz"
    np.savez_compressed(out, **payload)
    return out, audit


def main():
    family_gt = adp6.evaluator_data("train")["family_gt"]
    manifest = {
        "description": "ADP13 S4 evaluator-only diagnostic export; family_gt_id was not used during training.",
        "candidate_indexing": "zero-based: 0..4 correspond to C1..C5",
        "family_axis": "E0 train response-family order, N=1521",
        "cost_definition": "costs_rX[U,j] = E(U,j)",
        "rounds": ROUNDS.tolist(),
        "files": {},
    }
    for seed in range(3):
        path, audit = export_seed(seed, family_gt)
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "round_audit": audit,
        }
        print(f"exported {path}", flush=True)
    manifest_path = TARGET / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"pass": True, "target": str(TARGET), "files": list(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
