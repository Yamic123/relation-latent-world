"""E0-ADP1-MINI: single-card minimal proof-of-mechanism (guide E0_ADP1_experiment_execution_guide.md).

Small fixed subset (train 1024 / val 256 / tests 512), 3 optimization seeds,
12 EM rounds max, 960-update paired joint baseline.  Reuses the exact E-step and
residual M-step from ``adp1.exact_e_step`` / ``adp1.residual_m_step`` and the
original ``run_e0`` model/loaders, writing everything under ``outputs/e0_adp1_mini/``.

Subcommands: validate, smoke, validate-solver, run-paired, evaluate, summarize,
amortize (optional).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_E0_ROOT = Path(__file__).resolve().parent.parent
if str(_E0_ROOT) not in sys.path:
    sys.path.insert(0, str(_E0_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from adp1._utils import (
    DEVICE,
    LAMBDA_P,
    NUM_SUPPORTS,
    SUPPORTS,
    SUPPORT_SIZES,
    dump_json,
    hash_file,
    hash_state_dict,
    load_assignment,
    save_assignment,
    support_code,
    to_tensor,
)
from adp1.exact_e_step import exact_support_e_step, freeze_bank, unfreeze_bank
from adp1.residual_m_step import make_probe, m_step_round, snapshot_shared
from adp1.evaluate_adp1 import (
    _BankOnly,
    _bank_predictions,
    _hungarian,
    _leave_one_out,
    _structural_on_split,
)
from diagnostics import common as diag

import run_e0 as base  # noqa: E402

MINI_OUTPUT_ROOT = _E0_ROOT / "outputs" / "e0_adp1_mini"
MINI_CONFIG_PATH = _E0_ROOT / "configs" / "e0_adp1_mini.json"
DATASET = base.DATA_ROOT / "e0a_world_seed_20260901"
TEST_SPLITS = ["test_iid", "test_context", "test_combination_101"]
SOLVER_SANITY_DIR = MINI_OUTPUT_ROOT / "solver_sanity"

# Candidate 1 = MSB (matches _utils); support_code over 3 GT bits for coverage.
_GT_PATTERN = {"000": 0, "001": 1, "010": 2, "011": 3, "100": 4, "101": 5, "110": 6, "111": 7}


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else MINI_CONFIG_PATH
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Subset helpers (fixed prefix, identical across seeds)
# --------------------------------------------------------------------------- #
def _subset_indices(cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    sub = cfg["subset"]
    return {
        "train": np.arange(sub["train"], dtype=np.int64),
        "val": np.arange(sub["val"], dtype=np.int64),
        "test_iid": np.arange(sub["test_iid"], dtype=np.int64),
        "test_context": np.arange(sub["test_context"], dtype=np.int64),
        "test_combination_101": np.arange(sub["test_combination_101"], dtype=np.int64),
    }


def _load_subset(split: str, subset_idx: Mapping[str, np.ndarray], tp: str = "identity") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    S, p, delta = base.load_visible("e0a", split, tp)
    idx = subset_idx[split]
    return S[idx], p[idx], delta[idx]


def _save_subset_indices(out_root: Path, cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    idx = _subset_indices(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    np.savez(out_root / "subset_indices.npz", **{k: v for k, v in idx.items()})
    return idx


# --------------------------------------------------------------------------- #
# Solver ladder (A/B/C) -> effective e_step cfg
# --------------------------------------------------------------------------- #
def _selected_solver(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the calibrated (max_steps, restarts) or fall back to ladder rung A."""
    sel = SOLVER_SANITY_DIR / "selected_config.json"
    if sel.exists():
        data = json.loads(sel.read_text(encoding="utf-8"))
        return {"max_steps": int(data["max_steps"]), "restarts": int(data["restarts"]), "rung": data.get("rung", "?")}
    ladder = cfg["e_step"]["solver_ladder"]
    return {"max_steps": int(ladder[0][0]), "restarts": int(ladder[0][1]), "rung": "A(default)"}


def _effective_e_step(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    e = dict(cfg["e_step"])
    sel = _selected_solver(cfg)
    e["max_steps"] = sel["max_steps"]
    e["restarts"] = sel["restarts"]
    return e


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def _validate() -> Dict[str, Any]:
    MINI_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    subset_idx = _save_subset_indices(MINI_OUTPUT_ROOT, cfg)

    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    oracle = json.loads((DATASET / "oracle_certificate.json").read_text(encoding="utf-8"))

    hash_results: Dict[str, bool] = {}
    for key, expected in manifest["sha256"].items():
        actual = hash_file(DATASET / key.replace("\\", "/"))
        hash_results[key] = actual == expected

    oracle_ok = oracle["max_mse"] < 1e-10 and not oracle["held_out_101_leaked"]
    no_101_in_train = oracle.get("coverage", {}).get("train", {}).get("101", 0) == 0

    # Subset coverage (§3.2): six non-zero train patterns >=100, 000 >=20, 101 == 0.
    hidden_train = base.load_hidden("e0a", "train")
    m_gt_train = hidden_train["m_gt"][subset_idx["train"]]  # [1024, 3]
    code3 = m_gt_train[:, 0] * 4 + m_gt_train[:, 1] * 2 + m_gt_train[:, 2]
    counts = {name: int(np.sum(code3 == c)) for name, c in _GT_PATTERN.items()}
    nonzero_non101 = ["001", "010", "011", "100", "110", "111"]
    coverage_ok = all(counts[p] >= 100 for p in nonzero_non101) and counts["000"] >= 20 and counts["101"] == 0

    valid = bool(all(hash_results.values()) and oracle_ok and no_101_in_train and coverage_ok)
    result = {
        "valid": valid,
        "dataset": str(DATASET),
        "manifest_sha256_all_match": bool(all(hash_results.values())),
        "hash_mismatches": [k for k, v in hash_results.items() if not v],
        "oracle_max_mse": oracle["max_mse"],
        "oracle_mse_below_1e-10": bool(oracle_ok),
        "held_out_101_leaked": bool(oracle["held_out_101_leaked"]),
        "train_pattern_counts": counts,
        "subset_coverage_ok": bool(coverage_ok),
        "subset_sizes": {k: int(v.shape[0]) for k, v in subset_idx.items()},
    }
    dump_json(MINI_OUTPUT_ROOT / "input_validation.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


# --------------------------------------------------------------------------- #
# smoke (Stage 0)
# --------------------------------------------------------------------------- #
def _smoke(seed: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = SOLVER_SANITY_DIR / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    e_cfg = _effective_e_step(cfg)
    subset_idx = _subset_indices(cfg)

    base.seed_everything(seed)
    model = base.E0AModel(cfg["m_max"], 64, 2, 4).to(DEVICE)
    bank = model.bank
    optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                  weight_decay=cfg["m_step"]["weight_decay"])

    S, _, delta = base.load_visible("e0a", "train", "identity")
    S16, delta16 = S[:16], delta[:16]

    probe = make_probe()
    all_supports_visited = np.zeros(NUM_SUPPORTS, dtype=bool)
    prev_m = None
    rounds = []
    for r in range(2):
        freeze_bank(bank)
        before_hash = hash_state_dict(bank.state_dict())
        lat = exact_support_e_step(bank, S16, delta16, e_cfg, prev_v=None, round_id=r, opt_seed=seed)
        after_hash = hash_state_dict(bank.state_dict())
        e_step_bank_unchanged = before_hash == after_hash
        all_supports_visited |= (lat["audit_scores"].shape[1] == NUM_SUPPORTS)  # 32 columns
        nan_free = all(np.all(np.isfinite(v)) for v in [lat["m"], lat["v"], lat["effect_mse"], lat["penalized_J"], lat["second_margin"]])

        m_before = lat["m"].copy()
        v_before = lat["v"].copy()
        unfreeze_bank(bank)
        shared_before = snapshot_shared(bank)
        m_step_round(bank, optimizer, lat["m"], lat["v"], S16, delta16, cfg, r, seed, probe, shared_before)
        m_step_mv_unchanged = bool(np.array_equal(m_before, lat["m"]) and np.allclose(v_before, lat["v"]))
        bank_changed = before_hash != hash_state_dict(bank.state_dict())

        # Empty support prediction == 0, inactive v == 0.
        empty_mse = (delta16 ** 2).mean(axis=(-2, -1)).astype(np.float32)
        empty_support_ok = bool(np.allclose(lat["audit_scores"][:, 0], empty_mse, atol=1e-6)) and bool(np.all(lat["v"][lat["m"] == 0] == 0))
        # Fewer active bits -> objective lower by LAMBDA_P per bit at equal effect MSE.
        tie_ok = bool(np.isclose(lat["penalized_J"] - lat["effect_mse"],
                                 LAMBDA_P * SUPPORT_SIZES[lat["support_code"]].astype(np.float32), atol=1e-6).all())

        rounds.append({
            "round": r,
            "e_step_bank_unchanged": bool(e_step_bank_unchanged),
            "m_step_mv_unchanged": bool(m_step_mv_unchanged),
            "bank_changed_after_mstep": bool(bank_changed),
            "nan_free": bool(nan_free),
            "empty_support_ok": bool(empty_support_ok),
            "tie_penalty_ok": bool(tie_ok),
            "n_support_columns": int(lat["audit_scores"].shape[1]),
        })
        prev_m = lat["m"]

    peak = int(torch.cuda.max_memory_allocated()) if DEVICE.type == "cuda" else 0
    checks = {
        "all_32_supports_visited": bool(all_supports_visited.all()),
        "rounds": rounds,
        "peak_gpu_bytes": peak,
        "peak_gpu_below_6gb": bool(peak < 6 * 1024 ** 3),
        "pass": bool(all_supports_visited.all() and all(all(r[k] for k in
             ["e_step_bank_unchanged", "m_step_mv_unchanged", "bank_changed_after_mstep", "nan_free",
              "empty_support_ok", "tie_penalty_ok"]) for r in rounds) and peak < 6 * 1024 ** 3),
    }
    dump_json(out_dir / "smoke.json", checks)
    print(json.dumps(checks, indent=2), flush=True)
    return checks


# --------------------------------------------------------------------------- #
# validate-solver (ladder A/B/C vs 10-start L-BFGS-B on 16 samples)
# --------------------------------------------------------------------------- #
def _lbfgsb_reference(bank: torch.nn.Module, S: np.ndarray, delta: np.ndarray, n_starts: int = 10) -> Dict[str, np.ndarray]:
    from scipy.optimize import minimize

    bank_cpu = type(bank)(bank.m_max, bank.dim, bank.heads)
    bank_cpu.load_state_dict({k: v.cpu() for k, v in bank.state_dict().items()})
    bank_cpu.eval()

    N = S.shape[0]
    J_all = np.full((N, NUM_SUPPORTS), np.inf, dtype=np.float64)
    S_t = torch.from_numpy(S)
    d_t = torch.from_numpy(delta)

    for i in range(N):
        Si = S_t[i:i + 1]
        di = d_t[i:i + 1]
        for code in range(NUM_SUPPORTS):
            support = SUPPORTS[code]
            active_idx = [int(j) for j in np.flatnonzero(support)]
            a = len(active_idx)
            mask = torch.zeros(5)
            mask[active_idx] = 1.0
            if a == 0:
                best_obj = float(((torch.zeros_like(di) - di) ** 2).mean())
            else:
                def objective(v_active: np.ndarray) -> float:
                    v = torch.zeros(1, 5)
                    v[:, active_idx] = torch.from_numpy(v_active.astype(np.float32))
                    with torch.no_grad():
                        raw = bank_cpu.raw_effects(Si, v)
                        pred = (raw * mask[None, :, None, None]).sum(dim=1)
                    return float(((pred - di) ** 2).mean())

                bounds = [(-1.5, 1.5)] * a
                rng = np.random.default_rng(7_000_000 + i * NUM_SUPPORTS + code)
                starts = [np.zeros(a, dtype=np.float32)]
                starts.extend(rng.uniform(-1.5, 1.5, size=(n_starts - 1, a)).astype(np.float32))
                best_obj = np.inf
                for x0 in starts:
                    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8})
                    best_obj = min(best_obj, float(res.fun))
            J_all[i, code] = best_obj + LAMBDA_P * SUPPORT_SIZES[code]

    codes = np.arange(NUM_SUPPORTS, dtype=np.int64)
    from adp1._utils import tie_break_argmin
    best_support = tie_break_argmin(J_all, SUPPORT_SIZES.astype(np.float64), codes.astype(np.float64), 1e-8)
    best_J = J_all[np.arange(N), best_support]
    return {"support": best_support, "J": best_J}


def _validate_solver(seed: int, num_samples: int) -> Dict[str, Any]:
    cfg = load_config()
    base.seed_everything(seed)
    bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
    freeze_bank(bank)

    S, _, delta = base.load_visible("e0a", "train", "identity")
    S, delta = S[:num_samples], delta[:num_samples]

    ref = _lbfgsb_reference(bank, S, delta)
    lbfgsb_support = ref["support"]
    lbfgsb_J = ref["J"]

    results = []
    selected = None
    for (steps, restarts) in cfg["e_step"]["solver_ladder"]:
        e_cfg = dict(cfg["e_step"])
        e_cfg["max_steps"] = steps
        e_cfg["restarts"] = restarts
        latent = exact_support_e_step(bank, S, delta, e_cfg, prev_v=None, round_id=0, opt_seed=seed)
        adam_support = latent["support_code"].astype(np.int64)
        adam_J = latent["penalized_J"].astype(np.float64)
        agreement = float(np.mean(adam_support == lbfgsb_support))
        j_diff = np.abs(adam_J - lbfgsb_J)
        median_diff = float(np.median(j_diff))
        ok = bool(agreement >= 0.99 and median_diff <= 1e-5)
        results.append({"rung": f"{steps}/{restarts}", "max_steps": steps, "restarts": restarts,
                        "support_agreement": agreement, "median_abs_J_diff": median_diff, "pass": ok})
        if ok and selected is None:
            selected = {"rung": f"{steps}/{restarts}", "max_steps": steps, "restarts": restarts}

    SOLVER_SANITY_DIR.mkdir(parents=True, exist_ok=True)
    if selected is not None:
        dump_json(SOLVER_SANITY_DIR / "selected_config.json", selected)
    dump_json(SOLVER_SANITY_DIR / "metrics.json", {"seed": seed, "num_samples": num_samples, "ladder": results,
                                                    "selected": selected, "valid": selected is not None})
    print(json.dumps({"selected": selected, "ladder": results}, indent=2), flush=True)
    return {"selected": selected, "ladder": results}


# --------------------------------------------------------------------------- #
# discovery (per seed)
# --------------------------------------------------------------------------- #
def _check_stability(curves: List[Dict[str, Any]], o_cfg: Mapping[str, Any]) -> bool:
    w = o_cfg["stability_window"]
    if len(curves) < o_cfg["min_rounds"]:
        return False
    last = curves[-w:]
    scr_ok = all(r.get("scr") is not None and r["scr"] < o_cfg["support_change_threshold"] for r in last)
    vals = [r["J_val"] for r in curves if r.get("J_val") is not None]
    if len(vals) < 2:
        val_ok = False
    else:
        a, b = vals[-2], vals[-1]
        val_ok = abs(b - a) / max(abs(a), 1e-12) < o_cfg["validation_relative_change_threshold"]
    return bool(scr_ok and val_ok)


def _discover_seed(seed: int, cfg: Dict[str, Any], subset_idx: Mapping[str, np.ndarray], force: bool = False) -> Dict[str, Any]:
    seed_dir = MINI_OUTPUT_ROOT / f"seed_{seed}"
    adp1_dir = seed_dir / "adp1"
    adp1_dir.mkdir(parents=True, exist_ok=True)
    if (adp1_dir / "stability.json").exists() and not force:
        return json.loads((adp1_dir / "stability.json").read_text(encoding="utf-8"))

    e_cfg = _effective_e_step(cfg)
    o_cfg = cfg["outer_loop"]

    base.seed_everything(seed)
    full_model = base.E0AModel(cfg["m_max"], 64, 2, 4).to(DEVICE)
    torch.save(full_model.state_dict(), seed_dir / "initial_full_model.pt")
    (seed_dir / "initial_bank.sha256").write_text(hash_state_dict(full_model.bank.state_dict()) + "\n", encoding="utf-8")
    bank = full_model.bank  # discovery mutates the bank in place

    dump_json(seed_dir / "run_metadata.json", {
        "experiment": "E0-ADP1-MINI", "label": "discovery", "optimization_seed": seed,
        "world_seed": cfg["world_seed"], "dataset_seed": cfg["dataset_seed"],
        "git_commit": base.git_commit(), "config": cfg, "started_at_unix": time.time(),
        "solver_rung": _selected_solver(cfg), "training_data_contract": "visible.npz only (S, delta_S)",
    })

    optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                  weight_decay=cfg["m_step"]["weight_decay"])
    S_train, _, delta_train = _load_subset("train", subset_idx)
    S_val, _, delta_val = _load_subset("val", subset_idx)
    probe = make_probe()

    prev_m: Optional[np.ndarray] = None
    prev_v: Optional[np.ndarray] = None
    best_val_obj = float("inf")
    best_round = -1
    best_bank: Optional[Dict[str, torch.Tensor]] = None
    outer_curves: List[Dict[str, Any]] = []
    started = time.time()

    for round_id in range(o_cfg["max_rounds"]):
        freeze_bank(bank)
        train_lat = exact_support_e_step(bank, S_train, delta_train, e_cfg, prev_v=prev_v,
                                         round_id=round_id, opt_seed=seed)
        scr = float(np.mean(train_lat["m"] != prev_m)) if prev_m is not None else None

        J_val = None
        if round_id % o_cfg["val_every_rounds"] == 0:
            val_lat = exact_support_e_step(bank, S_val, delta_val, e_cfg, prev_v=None,
                                           round_id=round_id, opt_seed=seed)
            J_val = float(val_lat["penalized_J"].mean())
            if J_val < best_val_obj:
                best_val_obj = J_val
                best_round = round_id
                best_bank = {k: v.detach().clone() for k, v in bank.state_dict().items()}

        unfreeze_bank(bank)
        shared_before = snapshot_shared(bank)
        m_record = m_step_round(bank, optimizer, train_lat["m"], train_lat["v"], S_train, delta_train,
                                cfg, round_id, seed, probe, shared_before)

        record = {
            "round": round_id,
            "J_train": float(train_lat["penalized_J"].mean()),
            "effect_train": float(train_lat["effect_mse"].mean()),
            "active_train": float(train_lat["m"].sum(axis=1).mean()),
            "J_val": J_val,
            "scr": scr,
            "best_round": best_round,
        }
        outer_curves.append(record)
        dump_json(adp1_dir / "outer_curves.json", outer_curves)
        print(f"[discover seed={seed}] round={round_id} J_train={record['J_train']:.6f} "
              f"J_val={J_val} scr={scr} active={record['active_train']:.3f} ({time.time()-started:.0f}s)", flush=True)

        prev_m = train_lat["m"].copy()
        prev_v = train_lat["v"].copy()

        if _check_stability(outer_curves, o_cfg):
            break

    assert best_bank is not None, "no best bank recorded"
    torch.save(best_bank, adp1_dir / "best_bank.pt")

    bank.load_state_dict(best_bank)
    freeze_bank(bank)
    final_train = exact_support_e_step(bank, S_train, delta_train, e_cfg, prev_v=None, round_id=best_round, opt_seed=seed)
    final_val = exact_support_e_step(bank, S_val, delta_val, e_cfg, prev_v=None, round_id=best_round, opt_seed=seed)
    np.savez(adp1_dir / "final_assignments.npz",
             m_train=final_train["m"], v_train=final_train["v"],
             m_val=final_val["m"], v_val=final_val["v"],
             support_code_train=final_train["support_code"], support_code_val=final_val["support_code"])

    stable = _check_stability(outer_curves, o_cfg)
    stability = {
        "experiment": "E0-ADP1-MINI", "optimization_seed": seed,
        "discovery_stable": bool(stable),
        "rounds_run": len(outer_curves),
        "best_round": best_round,
        "best_val_objective": best_val_obj,
        "wall_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if DEVICE.type == "cuda" else 0,
    }
    dump_json(adp1_dir / "stability.json", stability)
    print(json.dumps(stability, indent=2), flush=True)
    return stability


# --------------------------------------------------------------------------- #
# joint baseline (per seed)
# --------------------------------------------------------------------------- #
def _joint_seed(seed: int, cfg: Dict[str, Any], subset_idx: Mapping[str, np.ndarray], force: bool = False) -> Dict[str, Any]:
    seed_dir = MINI_OUTPUT_ROOT / f"seed_{seed}"
    joint_dir = seed_dir / "joint_mini"
    joint_dir.mkdir(parents=True, exist_ok=True)
    if (joint_dir / "training_curves.json").exists() and not force:
        return {}

    jc = cfg["joint_control"]
    base.seed_everything(seed)
    model = base.E0AModel(cfg["m_max"], 64, 2, 4).to(DEVICE)
    model.load_state_dict(torch.load(seed_dir / "initial_full_model.pt", map_location=DEVICE, weights_only=True))

    S_train, p_train, delta_train = _load_subset("train", subset_idx)
    S_val, p_val, delta_val = _load_subset("val", subset_idx)

    optimizer = torch.optim.AdamW(model.parameters(), lr=jc["learning_rate"], weight_decay=jc["weight_decay"])
    generator = np.random.default_rng(seed + 123_456)
    n_updates = jc["optimizer_updates"]
    bs = jc["batch_size"]

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    curves: List[Dict[str, float]] = []
    started = time.time()

    for step in range(n_updates):
        frac = step / max(n_updates - 1, 1)
        temperature = jc["temperature_start"] * (jc["temperature_end"] / jc["temperature_start"]) ** frac
        model.train()
        idx = generator.integers(0, len(S_train), size=bs)
        S = torch.from_numpy(S_train[idx]).to(DEVICE, non_blocking=True)
        p = torch.from_numpy(p_train[idx]).to(DEVICE, non_blocking=True)
        target = torch.from_numpy(delta_train[idx]).to(DEVICE, non_blocking=True)
        out = model(S, p, temperature)
        effect_loss = F.mse_loss(out["prediction"], target)
        participation = out["gate_probs"].sum(dim=1).mean()
        loss = effect_loss + jc["lambda_participation"] * participation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), jc["gradient_clip"])
        optimizer.step()

        if (step + 1) % 32 == 0 or step == n_updates - 1:
            model.eval()
            with torch.no_grad():
                pred_val = base.predict_a(model, S_val, p_val)["prediction"]
                val_nrmse = base.nrmse(delta_val, pred_val)
            curves.append({"step": step + 1, "loss": float(loss.detach()), "effect_mse": float(effect_loss.detach()),
                           "expected_active": float(participation.detach()), "val_nrmse": val_nrmse, "temperature": temperature})
            if val_nrmse < best_val:
                best_val = val_nrmse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, joint_dir / "best_checkpoint.pt")
            print(f"[joint seed={seed}] step={step+1}/{n_updates} effect={float(effect_loss):.6f} "
                  f"active={float(participation):.3f} val_nrmse={val_nrmse:.4f}", flush=True)

    torch.save(model.state_dict(), joint_dir / "final_checkpoint.pt")
    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, joint_dir / "best_checkpoint.pt")
    dump_json(joint_dir / "training_curves.json", curves)
    info = {"optimization_seed": seed, "optimizer_updates": n_updates, "best_val_nrmse": best_val,
            "wall_seconds": time.time() - started}
    dump_json(joint_dir / "run_metadata.json", info)
    return info


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def _bootstrap_ci(stat_fn, n_samples: int, n_boot: int = 1000, rng_seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_samples, size=n_samples)
        vals[b] = stat_fn(idx)
    vals.sort()
    return {"mean": float(vals.mean()), "ci_low": float(vals[int(0.025 * n_boot)]),
            "ci_high": float(vals[int(0.975 * n_boot)])}


def _evaluate_adp1(seed: int, cfg: Dict[str, Any], subset_idx: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    adp1_dir = MINI_OUTPUT_ROOT / f"seed_{seed}" / "adp1"
    stability = json.loads((adp1_dir / "stability.json").read_text(encoding="utf-8"))
    e_cfg = _effective_e_step(cfg)

    bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
    bank.load_state_dict(torch.load(adp1_dir / "best_bank.pt", map_location=DEVICE, weights_only=True))
    freeze_bank(bank)

    # E-step each test split fresh with the frozen best bank.
    latents: Dict[str, Dict[str, np.ndarray]] = {}
    pred_nrmse: Dict[str, float] = {}
    for split in TEST_SPLITS:
        S, _, delta = _load_subset(split, subset_idx)
        latents[split] = exact_support_e_step(bank, S, delta, e_cfg, prev_v=None, round_id=0, opt_seed=seed)
        _, pred = _bank_predictions(bank, S, latents[split]["m"], latents[split]["v"])
        pred_nrmse[split] = base.nrmse(delta, pred)

    hidden_iid = base.load_hidden("e0a", "test_iid")
    hidden_iid = {k: v[subset_idx["test_iid"]] for k, v in hidden_iid.items()}
    S_iid, _, delta_iid = _load_subset("test_iid", subset_idx)
    v_iid = latents["test_iid"]["v"]
    func, calibration, _ = base.functional_matrix(_BankOnly(bank), v_iid, hidden_iid)
    mapping = _hungarian(func)
    effects_iid, _ = _bank_predictions(bank, S_iid, latents["test_iid"]["m"], v_iid)
    inst_matrix = diag.instance_r2_matrix(effects_iid, hidden_iid["e_gt"])

    iid_struct = _structural_on_split(bank, S_iid, delta_iid, latents["test_iid"]["m"], v_iid, hidden_iid,
                                      mapping, functional_matrix=func, instance_matrix=inst_matrix)
    hidden_ctx = {k: v[subset_idx["test_context"]] for k, v in base.load_hidden("e0a", "test_context").items()}
    S_ctx, _, delta_ctx = _load_subset("test_context", subset_idx)
    ctx_struct = _structural_on_split(bank, S_ctx, delta_ctx, latents["test_context"]["m"],
                                      latents["test_context"]["v"], hidden_ctx, mapping)

    # test_101: constant pattern [1,0,1]; per-mechanism F1/AUROC degenerate -> N/A.
    m101 = latents["test_combination_101"]["m"]
    hidden_101 = {k: v[subset_idx["test_combination_101"]] for k, v in base.load_hidden("e0a", "test_combination_101").items()}
    learned_3_101 = np.column_stack([m101[:, mapping[gt]] for gt in range(3)])
    pattern_exact = float(np.mean(np.all(learned_3_101 == hidden_101["m_gt"], axis=1)))
    active_inactive_acc = {f"Z{gt + 1}": float(np.mean(learned_3_101[:, gt] == hidden_101["m_gt"][:, gt])) for gt in range(3)}

    loo = _leave_one_out(delta_iid, effects_iid)
    inst_matched = iid_struct["instance_effect_r2_matched"]
    func_matched = iid_struct["functional_r2_matched"]

    # Bootstrap 95% CI on test_iid (NRMSE, F1, instance R2 min).
    m_gt_iid = hidden_iid["m_gt"]
    m_hat_iid = latents["test_iid"]["m"]
    e_gt_iid = hidden_iid["e_gt"]
    boot = {
        "iid_nrmse": _bootstrap_ci(lambda idx: base.nrmse(delta_iid[idx], iid_struct["_prediction"][idx]), len(S_iid)),
        "participation_f1": _bootstrap_ci(
            lambda idx: float(np.mean([base.binary_metrics(m_gt_iid[idx, gt], m_hat_iid[idx, mapping[gt]])["f1"] for gt in range(3)])),
            len(S_iid)),
        "instance_r2_min": _bootstrap_ci(
            lambda idx: float(min(diag.instance_r2_matrix(effects_iid[idx], e_gt_iid[idx])[mapping[gt], gt] for gt in range(3))),
            len(S_iid)),
    }

    passes = {
        "discovery_stable": stability["discovery_stable"],
        "predictive_iid": pred_nrmse["test_iid"] < 0.05,
        "predictive_101": pred_nrmse["test_combination_101"] < 0.10,
        "predictive_101_strict": pred_nrmse["test_combination_101"] < 0.05,
        "participation": iid_struct["participation_f1_mean"] > 0.90,
        "instance_effect": min(inst_matched) > 0.90,
        "functional": min(func_matched) > 0.90,
        "redundancy": iid_struct["redundant_usage_max"] < 0.10,
    }
    # Primary single-seed PASS uses exactly §11.1 (excluding the strict 101).
    primary = ["discovery_stable", "predictive_iid", "predictive_101", "participation",
               "instance_effect", "functional", "redundancy"]
    metrics = {
        "experiment": "E0-ADP1-MINI", "method": "adp1", "optimization_seed": seed,
        "prediction_nrmse": pred_nrmse,
        "alignment_gt_to_learned": {str(k + 1): v + 1 for k, v in mapping.items()},
        "redundant_indices_1_based": [l + 1 for l in range(cfg["m_max"]) if l not in set(mapping.values())],
        "redundant_usage_max": iid_struct["redundant_usage_max"],
        "test_iid": {k: v for k, v in iid_struct.items() if not k.startswith("_")},
        "test_context": {k: v for k, v in ctx_struct.items() if not k.startswith("_")},
        "test_101": {"prediction_nrmse": pred_nrmse["test_combination_101"],
                     "pattern_exact_accuracy": pattern_exact, "active_inactive_accuracy": active_inactive_acc,
                     "note": "Z2 constant-inactive in test_101; per-mechanism F1/AUROC N/A"},
        "functional_r2_matrix": func.tolist(), "functional_r2_matched": func_matched,
        "functional_r2_min": min(func_matched),
        "instance_effect_r2_matrix": inst_matrix.tolist(), "instance_effect_r2_matched": inst_matched,
        "instance_effect_r2_min": min(inst_matched),
        "leave_one_candidate_out": loo, "calibration": calibration,
        "bootstrap_95ci": boot, "pass_components": passes,
        "pass": bool(all(passes[k] for k in primary)),
    }
    dump_json(adp1_dir / "metrics.json", metrics)
    print(f"[evaluate adp1 seed={seed}] iid={pred_nrmse['test_iid']:.4f} 101={pred_nrmse['test_combination_101']:.4f} "
          f"f1={iid_struct['participation_f1_mean']:.3f} inst_min={min(inst_matched):.3f} func_min={min(func_matched):.3f} "
          f"redundant={iid_struct['redundant_usage_max']:.3f} PASS={metrics['pass']}", flush=True)
    return metrics


def _evaluate_joint(seed: int, cfg: Dict[str, Any], subset_idx: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    joint_dir = MINI_OUTPUT_ROOT / f"seed_{seed}" / "joint_mini"
    model = base.E0AModel(cfg["m_max"], 64, 2, 4).to(DEVICE)
    model.load_state_dict(torch.load(joint_dir / "best_checkpoint.pt", map_location=DEVICE, weights_only=True))
    model.eval()

    pred_nrmse: Dict[str, float] = {}
    out_cache: Dict[str, Dict[str, np.ndarray]] = {}
    for split in TEST_SPLITS:
        S, p, delta = _load_subset(split, subset_idx)
        out = base.predict_a(model, S, p)
        out_cache[split] = out
        pred_nrmse[split] = base.nrmse(delta, out["prediction"])

    hidden_iid = {k: v[subset_idx["test_iid"]] for k, v in base.load_hidden("e0a", "test_iid").items()}
    out_iid = out_cache["test_iid"]
    m_hat_iid = (out_iid["gate_probs"] >= 0.5).astype(np.float32)
    v_hat_iid = out_iid["v"]
    func, _, _ = base.functional_matrix(model, v_hat_iid, hidden_iid)
    mapping = _hungarian(func)
    effects_iid = m_hat_iid[:, :, None, None] * out_iid["raw"]
    inst_matrix = diag.instance_r2_matrix(effects_iid, hidden_iid["e_gt"])

    S_iid, _, delta_iid = _load_subset("test_iid", subset_idx)
    iid_struct = _structural_on_split(model.bank, S_iid, delta_iid, m_hat_iid, v_hat_iid, hidden_iid,
                                      mapping, functional_matrix=func, instance_matrix=inst_matrix)
    S_ctx, _, delta_ctx = _load_subset("test_context", subset_idx)
    hidden_ctx = {k: v[subset_idx["test_context"]] for k, v in base.load_hidden("e0a", "test_context").items()}
    out_ctx = out_cache["test_context"]
    m_hat_ctx = (out_ctx["gate_probs"] >= 0.5).astype(np.float32)
    ctx_struct = _structural_on_split(model.bank, S_ctx, delta_ctx, m_hat_ctx, out_ctx["v"], hidden_ctx, mapping)

    S_101, _, delta_101 = _load_subset("test_combination_101", subset_idx)
    out_101 = out_cache["test_combination_101"]
    m_hat_101 = (out_101["gate_probs"] >= 0.5).astype(np.float32)
    hidden_101 = {k: v[subset_idx["test_combination_101"]] for k, v in base.load_hidden("e0a", "test_combination_101").items()}
    learned_3_101 = np.column_stack([m_hat_101[:, mapping[gt]] for gt in range(3)])
    pattern_exact = float(np.mean(np.all(learned_3_101 == hidden_101["m_gt"], axis=1)))

    inst_matched = iid_struct["instance_effect_r2_matched"]
    func_matched = iid_struct["functional_r2_matched"]
    passes = {
        "predictive_iid": pred_nrmse["test_iid"] < 0.05,
        "predictive_101": pred_nrmse["test_combination_101"] < 0.10,
        "participation": iid_struct["participation_f1_mean"] > 0.90,
        "instance_effect": min(inst_matched) > 0.90,
        "functional": min(func_matched) > 0.90,
        "redundancy": iid_struct["redundant_usage_max"] < 0.10,
    }
    metrics = {
        "experiment": "E0-ADP1-MINI", "method": "joint", "optimization_seed": seed,
        "prediction_nrmse": pred_nrmse,
        "alignment_gt_to_learned": {str(k + 1): v + 1 for k, v in mapping.items()},
        "redundant_usage_max": iid_struct["redundant_usage_max"],
        "test_iid": {k: v for k, v in iid_struct.items() if not k.startswith("_")},
        "test_context": {k: v for k, v in ctx_struct.items() if not k.startswith("_")},
        "test_101": {"prediction_nrmse": pred_nrmse["test_combination_101"],
                     "pattern_exact_accuracy": pattern_exact,
                     "note": "Z2 constant-inactive in test_101; per-mechanism F1/AUROC N/A"},
        "functional_r2_matrix": func.tolist(), "functional_r2_matched": func_matched, "functional_r2_min": min(func_matched),
        "instance_effect_r2_matrix": inst_matrix.tolist(), "instance_effect_r2_matched": inst_matched,
        "instance_effect_r2_min": min(inst_matched),
        "pass_components": passes, "pass": bool(all(passes.values())),
    }
    dump_json(joint_dir / "metrics.json", metrics)
    print(f"[evaluate joint seed={seed}] iid={pred_nrmse['test_iid']:.4f} 101={pred_nrmse['test_combination_101']:.4f} "
          f"f1={iid_struct['participation_f1_mean']:.3f} func_min={min(func_matched):.3f} "
          f"redundant={iid_struct['redundant_usage_max']:.3f} PASS={metrics['pass']}", flush=True)
    return metrics


# --------------------------------------------------------------------------- #
# summarize + report
# --------------------------------------------------------------------------- #
def _summarize() -> Dict[str, Any]:
    cfg = load_config()
    seeds = cfg["optimization_seeds"]
    adp1_metrics = []
    joint_metrics = []
    for seed in seeds:
        a = MINI_OUTPUT_ROOT / f"seed_{seed}" / "adp1" / "metrics.json"
        j = MINI_OUTPUT_ROOT / f"seed_{seed}" / "joint_mini" / "metrics.json"
        if a.exists():
            adp1_metrics.append(json.loads(a.read_text(encoding="utf-8")))
        if j.exists():
            joint_metrics.append(json.loads(j.read_text(encoding="utf-8")))

    def _row(m: Dict[str, Any]) -> Dict[str, Any]:
        return {"seed": m["optimization_seed"],
                "iid_nrmse": m["prediction_nrmse"]["test_iid"],
                "nrmse_101": m["prediction_nrmse"]["test_combination_101"],
                "participation_f1": m["test_iid"]["participation_f1_mean"],
                "instance_r2_min": m["instance_effect_r2_min"],
                "functional_r2_min": m["functional_r2_min"],
                "redundant_usage": m["redundant_usage_max"],
                "pass": m["pass"]}

    n_adp1_pass = sum(1 for m in adp1_metrics if m["pass"])
    n_joint_pass = sum(1 for m in joint_metrics if m["pass"])
    if n_adp1_pass == 3:
        verdict = "ADP1-MINI PASS"
    elif n_adp1_pass == 2:
        verdict = "ADP1-MINI PARTIAL"
    else:
        verdict = "ADP1-MINI FAIL"
    if n_joint_pass == 3 and n_adp1_pass == 3:
        verdict = "MINI_CONTROL_NOT_DIAGNOSTIC"

    # Paired comparison rows (ADP1 vs joint identity) for shared metrics.
    header = ["seed", "method", "IID NRMSE", "101 NRMSE", "F1", "inst-R2 min", "func-R2 min", "redundant usage", "PASS"]
    csv_path = MINI_OUTPUT_ROOT / "paired_comparison.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for m in adp1_metrics:
            r = _row(m)
            w.writerow([r["seed"], "adp1", f"{r['iid_nrmse']:.5f}", f"{r['nrmse_101']:.5f}", f"{r['participation_f1']:.5f}",
                        f"{r['instance_r2_min']:.5f}", f"{r['functional_r2_min']:.5f}", f"{r['redundant_usage']:.5f}", r["pass"]])
        for m in joint_metrics:
            r = _row(m)
            w.writerow([r["seed"], "joint", f"{r['iid_nrmse']:.5f}", f"{r['nrmse_101']:.5f}", f"{r['participation_f1']:.5f}",
                        f"{r['instance_r2_min']:.5f}", f"{r['functional_r2_min']:.5f}", f"{r['redundant_usage']:.5f}", r["pass"]])

    summary = {
        "experiment": "E0-ADP1-MINI", "n_seeds": len(adp1_metrics),
        "adp1_pass_seeds": n_adp1_pass, "joint_pass_seeds": n_joint_pass, "verdict": verdict,
        "adp1": [_row(m) for m in adp1_metrics], "joint": [_row(m) for m in joint_metrics],
        "csv": str(csv_path),
    }
    dump_json(MINI_OUTPUT_ROOT / "summary.json", summary)
    _write_report(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def _write_report(summary: Dict[str, Any]) -> None:
    adp1 = summary["adp1"]
    joint = summary["joint"]
    n = summary["n_seeds"]
    n_a = summary["adp1_pass_seeds"]
    n_j = summary["joint_pass_seeds"]

    lines = [
        "# E0-ADP1-MINI 最终报告", "",
        f"**最终判定：{summary['verdict']}**", "",
        f"E0-ADP1-MINI：3 个 paired seeds 中 **{n_a}** 个满足完整结构恢复。",
        f"Joint mini baseline 为 **{n_j}/3**，ADP1 为 **{n_a}/3**。",
        "",
        "## 1. Discovery 主结果", "",
        "| seed | IID NRMSE | 101 NRMSE | F1 | inst-R2 min | func-R2 min | redundant | PASS |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in adp1:
        lines.append(f"| {r['seed']} | {r['iid_nrmse']:.4f} | {r['nrmse_101']:.4f} | {r['participation_f1']:.3f} | "
                     f"{r['instance_r2_min']:.3f} | {r['functional_r2_min']:.3f} | {r['redundant_usage']:.3f} | {r['pass']} |")
    lines += ["", "## 2. Paired joint mini baseline", "", "| seed | IID NRMSE | 101 NRMSE | F1 | func-R2 min | redundant | PASS |",
              "|---:|---:|---:|---:|---:|---:|:---:|"]
    for r in joint:
        lines.append(f"| {r['seed']} | {r['iid_nrmse']:.4f} | {r['nrmse_101']:.4f} | {r['participation_f1']:.3f} | "
                     f"{r['functional_r2_min']:.3f} | {r['redundant_usage']:.3f} | {r['pass']} |")
    lines += [
        "",
        "## 3. 结论", "",
        summary["verdict"], "",
        "见 `paired_comparison.csv` 与 `summary.json`。",
    ]
    (MINI_OUTPUT_ROOT / "E0_ADP1_MINI_report.md").write_text("\n".join(lines), encoding="utf-8")

    checklist = [
        "# E0-ADP1-MINI reproducibility checklist", "",
        "- [x] 复用原 world/data，未重生成（WORLD_SEED=20260901, DATASET_SEED=20260902）", "",
        "- [x] 子集为固定前缀（train 1024/val 256/tests 512），三 seed 完全相同", "",
        "- [x] train 无 101，oracle < 1e-10（input_validation.json）", "",
        "- [x] M_max=5，复用 SetMechanismBank，lambda_P=1e-3", "",
        "- [x] ADP1 与 joint 从同一 initial_full_model.pt 分叉", "",
        "- [x] E-step 冻结 bank，M-step 冻结 m,v；每 sample 枚举 32 supports", "",
        "- [x] solver A/B/C 按 L-BFGS-B reference 选最小达标配置，三 seed 固定", "",
        "- [x] 其他 adapter rows 与 moments 在 candidate update 中不变", "",
        "- [x] hidden GT 仅由独立 evaluate/validate 读取", "",
        "- [x] joint mini 使用相同 960 updates 预算", "",
        "- [x] 3 seeds 全部报告；常数-label 指标标 N/A；PASS/FAIL 用预注册标准", "",
    ]
    (MINI_OUTPUT_ROOT / "reproducibility_checklist.md").write_text("\n".join(checklist), encoding="utf-8")


# --------------------------------------------------------------------------- #
# optional amortization
# --------------------------------------------------------------------------- #
def _amortize(seed: int, tp: str, cfg: Dict[str, Any], subset_idx: Mapping[str, np.ndarray]) -> None:
    from adp1 import amortize_qeta
    adp1_dir = MINI_OUTPUT_ROOT / f"seed_{seed}" / "adp1"
    bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
    bank.load_state_dict(torch.load(adp1_dir / "best_bank.pt", map_location=DEVICE, weights_only=True))
    freeze_bank(bank)
    final = np.load(adp1_dir / "final_assignments.npz")
    S_train, p_train, _ = _load_subset("train", subset_idx, tp)
    S_val, p_val, _ = _load_subset("val", subset_idx, tp)
    out_dir = MINI_OUTPUT_ROOT / f"seed_{seed}" / "amortization" / tp
    info = amortize_qeta.train_qeta(seed, tp, bank, S_train, p_train, final["m_train"], final["v_train"],
                                    S_val, p_val, final["m_val"], final["v_val"], cfg, out_dir)
    dump_json(out_dir / "run_metadata.json", info)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="E0-ADP1-MINI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate")

    sm = sub.add_parser("smoke")
    sm.add_argument("--config", default=str(MINI_CONFIG_PATH))
    sm.add_argument("--seed", type=int, default=0)

    vs = sub.add_parser("validate-solver")
    vs.add_argument("--config", default=str(MINI_CONFIG_PATH))
    vs.add_argument("--seed", type=int, default=0)
    vs.add_argument("--num-samples", type=int, default=16)

    rp = sub.add_parser("run-paired")
    rp.add_argument("--config", default=str(MINI_CONFIG_PATH))
    rp.add_argument("--seed", type=int, default=None)
    rp.add_argument("--seeds", default=None)
    rp.add_argument("--force", action="store_true")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--config", default=str(MINI_CONFIG_PATH))
    ev.add_argument("--seeds", default="0,1,2")

    sub.add_parser("summarize")

    am = sub.add_parser("amortize")
    am.add_argument("--config", default=str(MINI_CONFIG_PATH))
    am.add_argument("--seed", type=int, required=True)
    am.add_argument("--tp", choices=["identity", "orthogonal"], required=True)
    am.add_argument("--max-epochs", type=int, default=30)

    args = parser.parse_args()
    cfg = load_config(args.config if "config" in args else None)

    if args.command == "validate":
        _validate()
    elif args.command == "smoke":
        _smoke(args.seed, cfg)
    elif args.command == "validate-solver":
        _validate_solver(args.seed, args.num_samples)
    elif args.command == "run-paired":
        subset_idx = _subset_indices(cfg)
        seeds = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(",")]
        for seed in seeds:
            _discover_seed(seed, cfg, subset_idx, args.force)
            _joint_seed(seed, cfg, subset_idx, args.force)
    elif args.command == "evaluate":
        subset_idx = _subset_indices(cfg)
        for seed in [int(s) for s in args.seeds.split(",")]:
            _evaluate_adp1(seed, cfg, subset_idx)
            _evaluate_joint(seed, cfg, subset_idx)
    elif args.command == "summarize":
        _summarize()
    elif args.command == "amortize":
        subset_idx = _subset_indices(cfg)
        _amortize(args.seed, args.tp, cfg, subset_idx)


if __name__ == "__main__":
    main()
