"""Unit tests for E0-ADP1 (guide 20): the 12 pre-registered correctness checks.

Run with ``python E0/adp1/unit_tests.py``.  Writes ``outputs/e0_adp1/unit_tests.json``
and exits non-zero if any test fails.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# Expose E0/ on sys.path (mirrors run_adp1.py bootstrap).
_E0_ROOT = Path(__file__).resolve().parent.parent
if str(_E0_ROOT) not in sys.path:
    sys.path.insert(0, str(_E0_ROOT))

import numpy as np
import torch

from adp1._utils import (
    ADP1_OUTPUT_ROOT,
    DEVICE,
    LAMBDA_P,
    NUM_SUPPORTS,
    SUPPORTS,
    SUPPORT_SIZES,
    decode_support,
    hash_state_dict,
    load_assignment,
    save_assignment,
    support_code,
)
from adp1.exact_e_step import exact_support_e_step, freeze_bank, unfreeze_bank
from adp1.residual_m_step import m_step_round, snapshot_shared
from adp1.evaluate_adp1 import _hungarian
from adp1 import run_adp1

import run_e0 as base  # noqa: E402

TEST_DIM = 16
TEST_HEADS = 4


def _small_bank(seed: int = 0) -> torch.nn.Module:
    base.seed_everything(seed)
    return base.SetMechanismBank(5, TEST_DIM, TEST_HEADS).to(DEVICE)


def _test_e_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ec = dict(cfg["e_step"])
    ec["restarts"] = 3
    ec["max_steps"] = 30
    ec["chunk_size"] = 64
    return ec


def _small_data(n: int = 16, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    S = rng.uniform(-1.0, 1.0, size=(n, 3, 2)).astype(np.float32)
    delta = rng.uniform(-1.0, 1.0, size=(n, 3, 2)).astype(np.float32)
    return S, delta


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_support_enumeration() -> Tuple[bool, str]:
    assert SUPPORTS.shape == (32, 5), SUPPORTS.shape
    assert np.array_equal(decode_support(0), [0, 0, 0, 0, 0])
    assert np.array_equal(decode_support(31), [1, 1, 1, 1, 1])
    # Bit order: candidate 1 is MSB (weight 16), candidate 5 is LSB (weight 1).
    assert np.array_equal(decode_support(16), [1, 0, 0, 0, 0])
    assert np.array_equal(decode_support(1), [0, 0, 0, 0, 1])
    assert np.array_equal(SUPPORT_SIZES, SUPPORTS.sum(axis=1))
    for c in range(32):
        assert np.array_equal(support_code(decode_support(c)[None]), [c])
        assert support_code(SUPPORTS[c:c + 1])[0] == c
    return True, f"{NUM_SUPPORTS} supports, MSB order, popcounts verified"


def test_penalty(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    bank = _small_bank(1)
    freeze_bank(bank)
    S, delta = _small_data()
    latent = exact_support_e_step(bank, S, delta, _test_e_cfg(cfg), round_id=0, opt_seed=7)
    size = SUPPORT_SIZES[latent["support_code"]]
    expected = latent["effect_mse"] + LAMBDA_P * size.astype(np.float32)
    ok = bool(np.allclose(latent["penalized_J"], expected, atol=1e-6))
    return ok, f"J == effect_mse + {LAMBDA_P}*|m| (max |diff|={np.max(np.abs(latent['penalized_J'] - expected)):.2e})"


def test_empty_support(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    bank = _small_bank(2)
    freeze_bank(bank)
    S, delta = _small_data()
    latent = exact_support_e_step(bank, S, delta, _test_e_cfg(cfg), round_id=0, opt_seed=7)
    empty_mse = (delta ** 2).mean(axis=(-2, -1)).astype(np.float32)
    # Empty support (00000): prediction == 0, so effect_mse == mean(delta^2) with no penalty.
    ok_scores = bool(np.allclose(latent["audit_scores"][:, 0], empty_mse, atol=1e-6))
    # Inactive candidates carry zero v in the returned assignment (v_all zero-init).
    inactive_zero = bool(np.all(latent["v"][latent["m"] == 0] == 0))
    return ok_scores and inactive_zero, "empty support -> prediction 0 (no penalty), inactive v == 0"


def test_freeze_unfreeze() -> Tuple[bool, str]:
    bank = _small_bank(3)
    freeze_bank(bank)
    f_ok = (not bank.training) and all(not p.requires_grad for p in bank.parameters())
    unfreeze_bank(bank)
    u_ok = bank.training and all(p.requires_grad for p in bank.parameters())
    return f_ok and u_ok, "freeze sets eval + requires_grad False; unfreeze restores"


def test_e_step_freezes_bank(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    bank = _small_bank(4)
    freeze_bank(bank)
    before = hash_state_dict(bank.state_dict())
    S, delta = _small_data()
    exact_support_e_step(bank, S, delta, _test_e_cfg(cfg), round_id=0, opt_seed=8)
    after = hash_state_dict(bank.state_dict())
    return before == after, "E-step leaves bank state_dict bit-identical"


def test_adapter_isolation(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    bank = _small_bank(5)
    base.seed_everything(5)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                  weight_decay=cfg["m_step"]["weight_decay"])
    # Only candidate 2 has any active sample -> candidates {0,1,3,4} are skipped.
    m = np.zeros((64, 5), dtype=np.float32)
    m[:, 2] = 1.0
    v = np.zeros((64, 5), dtype=np.float32)
    S, delta = _small_data(64, 99)
    before = {k: vv.detach().clone() for k, vv in bank.state_dict().items()}
    record = m_step_round(bank, optimizer, m, v, S, delta, cfg, round_id=0, seed=5)
    adapters_before = before["adapters"]
    adapters_after = bank.state_dict()["adapters"]
    untouched = [j for j in range(5) if j != 2]
    rows_ok = all(torch.equal(adapters_after[j], adapters_before[j]) for j in untouched)
    target_changed = not torch.equal(adapters_after[2], adapters_before[2])
    skipped = {c["candidate"] for c in record["candidates"] if c.get("skipped_empty_candidate")}
    skip_ok = skipped == {0, 1, 3, 4}
    return rows_ok and target_changed and skip_ok, (
        f"non-target rows bit-identical, target changed, skipped={sorted(skipped)}"
    )


def test_residual_identity(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    bank = _small_bank(6)
    base.seed_everything(6)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                  weight_decay=cfg["m_step"]["weight_decay"])
    m = np.zeros((128, 5), dtype=np.float32)
    m[:, 2] = 1.0
    v = np.zeros((128, 5), dtype=np.float32)
    S, delta = _small_data(128, 100)
    freeze_bank(bank)
    with torch.no_grad():
        raw0 = bank.raw_effects(torch.from_numpy(S).to(DEVICE), torch.from_numpy(v).to(DEVICE))
        init_mse = float(torch.mean((raw0[:, 2] - torch.from_numpy(delta).to(DEVICE)) ** 2))
    unfreeze_bank(bank)
    m_step_round(bank, optimizer, m, v, S, delta, cfg, round_id=0, seed=6)
    freeze_bank(bank)
    with torch.no_grad():
        raw1 = bank.raw_effects(torch.from_numpy(S).to(DEVICE), torch.from_numpy(v).to(DEVICE))
        final_mse = float(torch.mean((raw1[:, 2] - torch.from_numpy(delta).to(DEVICE)) ** 2))
    ok = final_mse < init_mse
    return ok, f"sole active candidate fits delta directly (MSE {init_mse:.4f} -> {final_mse:.4f})"


def test_hidden_label_isolation() -> Tuple[bool, str]:
    adp1_dir = Path(__file__).resolve().parent
    train_files = ["_utils.py", "exact_e_step.py", "residual_m_step.py", "amortize_qeta.py"]
    offenders = []
    for name in train_files:
        text = (adp1_dir / name).read_text(encoding="utf-8")
        if "load_hidden" in text or "hidden_gt" in text:
            offenders.append(name)
    run_text = (adp1_dir / "run_adp1.py").read_text(encoding="utf-8")
    if "load_hidden" in run_text:
        offenders.append("run_adp1.py")
    ok = not offenders
    return ok, "discovery/amortization source never references hidden GT" if ok else f"offenders: {offenders}"


def test_permutation_equivalence() -> Tuple[bool, str]:
    matrix = np.asarray([[0.1, 0.9, 0.2], [0.8, 0.1, 0.3], [0.2, 0.3, 0.7]], dtype=np.float64)
    mapping = _hungarian(matrix)
    # Column argmax per row: row0->col1, row1->col0, row2->col2 (Hungarian on -matrix).
    expected = {0: 1, 1: 0, 2: 2}
    ok = mapping == expected
    return ok, f"Hungarian alignment on synthetic matrix -> {mapping}"


def test_nrmse_parity() -> Tuple[bool, str]:
    rng = np.random.default_rng(42)
    target = rng.normal(size=(50, 3, 2)).astype(np.float32)
    pred = target + 0.1 * rng.normal(size=(50, 3, 2)).astype(np.float32)
    num = np.sqrt(np.mean(np.sum((target - pred) ** 2, axis=(-2, -1))))
    den = np.sqrt(np.mean(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=(-2, -1))))
    manual = float(num / max(den, 1e-12))
    ok = bool(abs(base.nrmse(target, pred) - manual) < 1e-6)
    return ok, f"nrmse matches manual formula ({base.nrmse(target, pred):.6f} vs {manual:.6f})"


def test_seed_replay(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    bank = _small_bank(9)
    freeze_bank(bank)
    S, delta = _small_data(32, 111)
    r1 = exact_support_e_step(bank, S, delta, _test_e_cfg(cfg), round_id=0, opt_seed=21)
    r2 = exact_support_e_step(bank, S, delta, _test_e_cfg(cfg), round_id=0, opt_seed=21)
    m_ok = bool(np.array_equal(r1["m"], r2["m"]))
    v_ok = bool(np.allclose(r1["v"], r2["v"], atol=1e-6))
    j_ok = bool(np.allclose(r1["penalized_J"], r2["penalized_J"], atol=1e-5))
    return m_ok and v_ok and j_ok, "same seed -> identical init, assignment and objective"


def test_resume(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    # (a) assignment I/O round-trip.
    rng = np.random.default_rng(0)
    m = (rng.random((8, 5)) > 0.5).astype(np.uint8)
    v = rng.uniform(-1.5, 1.5, size=(8, 5)).astype(np.float32)
    tmp = ADP1_OUTPUT_ROOT / "_unit_tmp_assignment.npz"
    save_assignment(tmp, m, v, np.zeros(8), np.zeros(8), np.zeros(8))
    loaded = load_assignment(tmp)
    io_ok = bool(np.array_equal(loaded["m"], m) and np.allclose(loaded["v"], v))
    tmp.unlink(missing_ok=True)

    # (b) resume short-circuit: an existing stability.json is returned untouched.
    seed = 999_999
    done_dir = ADP1_OUTPUT_ROOT / f"seed_{seed}" / "discovery"
    done_dir.mkdir(parents=True, exist_ok=True)
    sentinel = {"discovery_stable": True, "sentinel": seed}
    (done_dir / "stability.json").write_text(json.dumps(sentinel), encoding="utf-8")
    result = run_adp1.discover_seed(seed, cfg)
    resume_ok = (result == sentinel)
    shutil.rmtree(ADP1_OUTPUT_ROOT / f"seed_{seed}", ignore_errors=True)

    return io_ok and resume_ok, "assignment round-trip + stability.json resume short-circuit"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main() -> int:
    cfg = run_adp1.load_config()
    tests: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
        ("support_enumeration", test_support_enumeration),
        ("penalty", lambda: test_penalty(cfg)),
        ("empty_support", lambda: test_empty_support(cfg)),
        ("freeze_unfreeze", test_freeze_unfreeze),
        ("e_step_freezes_bank", lambda: test_e_step_freezes_bank(cfg)),
        ("adapter_isolation", lambda: test_adapter_isolation(cfg)),
        ("residual_identity", lambda: test_residual_identity(cfg)),
        ("hidden_label_isolation", test_hidden_label_isolation),
        ("permutation_equivalence", test_permutation_equivalence),
        ("nrmse_parity", test_nrmse_parity),
        ("seed_replay", lambda: test_seed_replay(cfg)),
        ("resume", lambda: test_resume(cfg)),
    ]

    results: List[Dict[str, Any]] = []
    all_pass = True
    for name, fn in tests:
        try:
            passed, detail = fn()
        except Exception as exc:  # noqa: BLE001
            passed, detail = False, f"exception: {type(exc).__name__}: {exc}"
        all_pass = all_pass and passed
        results.append({"test": name, "pass": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    summary = {
        "experiment": "E0-ADP1",
        "n_tests": len(results),
        "n_pass": sum(1 for r in results if r["pass"]),
        "all_pass": all_pass,
        "results": results,
    }
    ADP1_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (ADP1_OUTPUT_ROOT / "unit_tests.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{summary['n_pass']}/{summary['n_tests']} unit tests passed -> "
          f"{ADP1_OUTPUT_ROOT / 'unit_tests.json'}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
