from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import scipy
import torch
from torch import Tensor


DIAGNOSTICS_DIR = Path(__file__).resolve().parent
E0_ROOT = DIAGNOSTICS_DIR.parent
REPO_ROOT = E0_ROOT.parent
if str(E0_ROOT) not in sys.path:
    sys.path.insert(0, str(E0_ROOT))

import run_e0 as base  # noqa: E402


OUTPUT_ROOT = E0_ROOT / "outputs" / "e0_diagnostics"
DEVICE = base.DEVICE


class RunLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_config(path: Path, value: Mapping[str, Any]) -> None:
    # JSON is a strict subset of YAML and avoids adding a PyYAML dependency.
    dump_json(path, dict(value))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def environment() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def seed_record(optimization_seed: int) -> Dict[str, int]:
    return {
        "world_seed": base.WORLD_SEED,
        "dataset_seed": base.DATASET_SEED,
        "optimization_seed": optimization_seed,
    }


def metadata(experiment: str, optimization_seed: int, config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "experiment": experiment,
        **seed_record(optimization_seed),
        "git_commit": git_commit(),
        "environment": environment(),
        "config": dict(config),
        "started_at_unix": time.time(),
        "diagnostic_only": True,
        "original_e0a_decision_unchanged": "E0 FAIL",
    }


def slices(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(n, start + batch_size))


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = float(np.sum((target - prediction) ** 2))
    centered = target - target.mean(axis=0, keepdims=True)
    denominator = float(np.sum(centered**2))
    return 1.0 - numerator / max(denominator, 1e-12)


def nrmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return base.nrmse(target, prediction)


def binary_metrics(target: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    return base.binary_metrics(target, score)


def load_all(split: str, tp: str = "identity") -> Dict[str, np.ndarray]:
    S, p, delta = base.load_visible("e0a", split, tp)
    hidden = base.load_hidden("e0a", split)
    return {"S": S, "p": p, "delta_S": delta, **hidden}


def gt_effects_torch(S: Tensor, m: Tensor, v: Tensor, world: Mapping[str, Tensor]) -> Tensor:
    linear = torch.einsum("jkab,nkb->njka", world["A"], S)
    value = (
        v[:, :, None, None] * linear
        + v[:, :, None, None] * world["b"][None]
        + 0.25 * v[:, :, None, None].square() * world["c"][None]
    )
    return m[:, :, None, None] * world["C"][None, :, :, None] * value


def world_tensors() -> Dict[str, Tensor]:
    return {
        key: torch.from_numpy(value).to(DEVICE)
        for key, value in base.load_world("e0a").items()
    }


def instance_r2_matrix(predicted_effects: np.ndarray, gt_effects: np.ndarray) -> np.ndarray:
    matrix = np.empty((predicted_effects.shape[1], gt_effects.shape[1]), dtype=np.float64)
    for learned in range(predicted_effects.shape[1]):
        for gt in range(gt_effects.shape[1]):
            matrix[learned, gt] = r2(gt_effects[:, gt], predicted_effects[:, learned])
    return matrix


def aggregate_seed_metrics(metrics: List[Mapping[str, Any]], scalar_paths: Mapping[str, List[str]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "n_seeds": len(metrics),
        "passed_seeds": int(sum(bool(item.get("pass")) for item in metrics)),
        "seed_metrics": metrics,
    }
    for name, path in scalar_paths.items():
        values = []
        for item in metrics:
            current: Any = item
            for key in path:
                current = current[key]
            values.append(float(current))
        result[name + "_mean"] = float(np.mean(values))
        result[name + "_std"] = float(np.std(values))
    return result
