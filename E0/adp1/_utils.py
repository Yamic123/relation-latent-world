"""Shared internal utilities for the E0-ADP1 diagnostic experiment.

This package reuses ``run_e0`` (the original E0-A learner) as ``base`` and
writes all of its own artifacts under ``E0/outputs/e0_adp1/`` (never touching
``e0a/main``, ``e0_diagnostics`` or the dataset).  The helpers here are the
small deterministic pieces (support bit decoding, assignment I/O, state-dict
hashing, seeded restarts, tie-break argmin) shared across the ADP1 modules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

E0_ROOT = Path(__file__).resolve().parent.parent
if str(E0_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(E0_ROOT))

import run_e0 as base  # noqa: E402

REPO_ROOT = E0_ROOT.parent
ADP1_ROOT = E0_ROOT / "adp1"
ADP1_OUTPUT_ROOT = E0_ROOT / "outputs" / "e0_adp1"
CONFIG_PATH = E0_ROOT / "configs" / "e0_adp1.json"
DEVICE = base.DEVICE

M_MAX = 5
NUM_SUPPORTS = 32
LAMBDA_P = 1e-3

# Support bit order: candidate 1 is the most-significant bit (guide 7.2).
# support_code = sum_j m[j] * 2 ** (4 - j); code 0 == 00000, code 31 == 11111.
_BIT_WEIGHTS = np.asarray([16, 8, 4, 2, 1], dtype=np.int64)


def decode_support(code: int) -> np.ndarray:
    """Return the 5-bit binary support vector (uint8) for an integer code."""
    bits = np.zeros(M_MAX, dtype=np.uint8)
    for j in range(M_MAX):
        bits[j] = (code >> (M_MAX - 1 - j)) & 1
    return bits


def support_code(m: np.ndarray) -> np.ndarray:
    """Encode an [...,5] binary array into integer support codes."""
    return (m.astype(np.int64) * _BIT_WEIGHTS).sum(axis=-1)


SUPPORTS = np.stack([decode_support(c) for c in range(NUM_SUPPORTS)], axis=0)  # [32,5]
SUPPORT_SIZES = SUPPORTS.sum(axis=1)  # [32]


def mix_seed(*values: int) -> int:
    """Deterministic integer mixing (no reliance on PYTHONHASHSEED)."""
    h = 0
    for x in values:
        h = (h * 1000003) ^ int(x)
    return h & 0x7FFFFFFF


def tie_break_argmin(scores: np.ndarray, sizes: np.ndarray, codes: np.ndarray, tie_eps: float) -> np.ndarray:
    """Per-row argmin with the fixed tie-break (smaller support, then smaller code).

    ``scores`` is [N, 32]; returns best code index [N].  ``sizes``/``codes`` are
    the per-column popcount and bit code (both ascending in column order).
    """
    min_score = scores.min(axis=1, keepdims=True)
    near = scores <= min_score + tie_eps
    out = np.empty(scores.shape[0], dtype=np.int64)
    for i in range(scores.shape[0]):
        cand = np.flatnonzero(near[i])
        # size primary (x1000 > max code 31), code secondary (ascending).
        out[i] = cand[np.argmin(sizes[cand] * 1000 + codes[cand])]
    return out


def hash_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(np.ascontiguousarray(state_dict[key].detach().cpu().float().numpy()).tobytes())
    return digest.hexdigest()


def hash_file(path: Path) -> str:
    return base.sha256_file(path)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def save_assignment(path: Path, m: np.ndarray, v: np.ndarray, effect_mse: np.ndarray,
                    penalized_J: np.ndarray, second_margin: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        m=m.astype(np.uint8),
        v=v.astype(np.float32),
        effect_mse=effect_mse.astype(np.float32),
        penalized_J=penalized_J.astype(np.float32),
        second_margin=second_margin.astype(np.float32),
        support_code=support_code(m).astype(np.uint8),
    )


def load_assignment(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(DEVICE, non_blocking=True)


def git_commit() -> str:
    return base.git_commit()
