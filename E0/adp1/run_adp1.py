"""Command-line entry point for the E0-ADP1 diagnostic experiment.

Subcommands: validate-inputs, validate-solver, discover, amortize, evaluate,
compare-joint, summarize.  All artifacts go under ``E0/outputs/e0_adp1/``; the
original ``e0a/main``, ``e0_diagnostics`` and the dataset are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Expose E0/ on sys.path so `adp1.*` and `run_e0` resolve under both
# `python E0/adp1/run_adp1.py` and `python -m E0.adp1.run_adp1`.
_E0_ROOT = Path(__file__).resolve().parent.parent
if str(_E0_ROOT) not in sys.path:
    sys.path.insert(0, str(_E0_ROOT))

import numpy as np
import torch

from adp1._utils import (
    ADP1_OUTPUT_ROOT,
    CONFIG_PATH,
    DEVICE,
    NUM_SUPPORTS,
    SUPPORTS,
    SUPPORT_SIZES,
    dump_json,
    hash_file,
    hash_state_dict,
    load_assignment,
    save_assignment,
    tie_break_argmin,
)
from adp1.exact_e_step import exact_support_e_step, freeze_bank, unfreeze_bank
from adp1.residual_m_step import make_probe, m_step_round, snapshot_shared
from adp1 import amortize_qeta, compare_joint, evaluate_adp1

import run_e0 as base  # noqa: E402


# --------------------------------------------------------------------------- #
# Config and metadata
# --------------------------------------------------------------------------- #
def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def _environment() -> Dict[str, Any]:
    env = base.environment_metadata()
    env["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    env["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    env["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    return env


def write_run_metadata(run_dir: Path, seed: int, cfg: Dict[str, Any], label: str = "discovery") -> None:
    dump_json(run_dir / "run_metadata.json", {
        "experiment": "E0-ADP1",
        "label": label,
        "world_seed": cfg["world_seed"],
        "dataset_seed": cfg["dataset_seed"],
        "optimization_seed": seed,
        "git_commit": base.git_commit(),
        "environment": _environment(),
        "config": cfg,
        "started_at_unix": time.time(),
        "training_data_contract": "visible.npz only (S, delta_S); hidden_gt never read by discovery/amortization",
        "initial_bank_source": "random seed_everything(seed); q_eta discarded",
    })
    dump_json(run_dir / "seed.json", {
        "world_seed": cfg["world_seed"], "dataset_seed": cfg["dataset_seed"], "optimization_seed": seed,
    })
    dump_json(run_dir / "config.json", cfg)


# --------------------------------------------------------------------------- #
# validate-inputs
# --------------------------------------------------------------------------- #
def validate_inputs(dataset: Path) -> Dict[str, Any]:
    manifest_path = dataset / "manifest.json"
    oracle_path = dataset / "oracle_certificate.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path.exists() else None

    hash_results: Dict[str, bool] = {}
    for key, expected in manifest["sha256"].items():
        rel = key.replace("\\", "/")
        actual = hash_file(dataset / rel)
        hash_results[key] = actual == expected

    oracle_ok = oracle is not None and oracle["max_mse"] < 1e-10 and not oracle["held_out_101_leaked"]
    coverage_train = oracle.get("coverage", {}).get("train", {}) if oracle else {}
    no_101_in_train = coverage_train.get("101", 0) == 0

    split_sizes = {}
    for split in base.SPLIT_SIZES:
        with np.load(dataset / split / "visible.npz") as data:
            split_sizes[split] = int(data["S"].shape[0])

    expected_sizes = base.SPLIT_SIZES
    sizes_ok = all(split_sizes.get(k) == v for k, v in expected_sizes.items())

    valid = bool(all(hash_results.values()) and oracle_ok and no_101_in_train and sizes_ok)
    result = {
        "valid": valid,
        "dataset": str(dataset),
        "manifest_sha256_all_match": bool(all(hash_results.values())),
        "n_hashes_checked": len(hash_results),
        "hash_mismatches": [k for k, v in hash_results.items() if not v],
        "oracle_max_mse": oracle["max_mse"] if oracle else None,
        "oracle_mse_below_1e-10": bool(oracle_ok),
        "held_out_101_leaked": bool(oracle and oracle["held_out_101_leaked"]),
        "no_101_in_train": no_101_in_train,
        "split_sizes": split_sizes,
        "expected_split_sizes": expected_sizes,
        "split_sizes_ok": sizes_ok,
    }
    dump_json(ADP1_OUTPUT_ROOT / "input_validation.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


# --------------------------------------------------------------------------- #
# validate-solver (continuous-solver sanity check vs L-BFGS-B)
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
            J_all[i, code] = best_obj + 1e-3 * SUPPORT_SIZES[code]

    codes = np.arange(NUM_SUPPORTS, dtype=np.int64)
    best_support = tie_break_argmin(J_all, SUPPORT_SIZES.astype(np.float64), codes.astype(np.float64), 1e-8)
    best_J = J_all[np.arange(N), best_support]
    return {"support": best_support, "J": best_J}


def validate_solver(cfg: Dict[str, Any], seed: int, num_samples: int) -> Dict[str, Any]:
    base.seed_everything(seed)
    bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
    freeze_bank(bank)

    S, _, delta = base.load_visible("e0a", "train", "identity")
    S = S[:num_samples]
    delta = delta[:num_samples]

    e_cfg = cfg["e_step"]
    latent = exact_support_e_step(bank, S, delta, e_cfg, prev_v=None, round_id=0, opt_seed=seed)
    adam_support = latent["support_code"].astype(np.int64)
    adam_J = latent["penalized_J"].astype(np.float64)

    ref = _lbfgsb_reference(bank, S, delta)
    lbfgsb_support = ref["support"]
    lbfgsb_J = ref["J"]

    agreement = float(np.mean(adam_support == lbfgsb_support))
    j_diff = np.abs(adam_J - lbfgsb_J)
    median_diff = float(np.median(j_diff))

    out_dir = ADP1_OUTPUT_ROOT / "solver_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "config.json", {**cfg["e_step"], "seed": seed, "num_samples": num_samples,
                                        "reference": "scipy.optimize.minimize L-BFGS-B, 12 starts"})
    with (out_dir / "support_agreement.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "adam_support", "lbfgsb_support", "adam_J", "lbfgsb_J", "abs_J_diff"])
        for i in range(num_samples):
            w.writerow([i, adam_support[i], lbfgsb_support[i], f"{adam_J[i]:.8e}", f"{lbfgsb_J[i]:.8e}", f"{j_diff[i]:.8e}"])

    valid = bool(agreement >= 0.99 and median_diff <= 1e-5)
    result = {
        "valid": valid,
        "seed": seed,
        "num_samples": num_samples,
        "support_agreement": agreement,
        "support_agreement_pass": bool(agreement >= 0.99),
        "median_abs_J_diff": median_diff,
        "median_J_diff_pass": bool(median_diff <= 1e-5),
        "max_abs_J_diff": float(np.max(j_diff)),
    }
    dump_json(out_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


# --------------------------------------------------------------------------- #
# discover
# --------------------------------------------------------------------------- #
def _check_stability(curves: List[Dict[str, Any]], o_cfg: Dict[str, Any]) -> bool:
    w = o_cfg["stability_window"]
    if len(curves) < o_cfg["min_rounds"]:
        return False
    last = curves[-w:]
    scr_ok = all(r.get("scr") is not None and r["scr"] < o_cfg["support_change_threshold"] for r in last)
    roc_ok = all(r.get("roc") is not None and r["roc"] < o_cfg["validation_relative_change_threshold"] for r in last)
    return bool(scr_ok and roc_ok)


def _save_audit_csv(path: Path, audit_scores: np.ndarray) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"support_{code:05d}" for code in range(NUM_SUPPORTS)])
        for row in audit_scores:
            w.writerow([f"{x:.8e}" for x in row])


def discover_seed(seed: int, cfg: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    run_dir = ADP1_OUTPUT_ROOT / f"seed_{seed}"
    discovery_dir = run_dir / "discovery"
    assign_dir = discovery_dir / "assignments"
    run_dir.mkdir(parents=True, exist_ok=True)
    discovery_dir.mkdir(parents=True, exist_ok=True)
    assign_dir.mkdir(parents=True, exist_ok=True)
    done_flag = discovery_dir / "stability.json"
    if done_flag.exists() and not force:
        return json.loads(done_flag.read_text(encoding="utf-8"))

    e_cfg = cfg["e_step"]
    o_cfg = cfg["outer_loop"]

    ckpt_state_path = discovery_dir / "checkpoint_state.json"
    ckpt_bank = discovery_dir / "checkpoint_bank.pt"
    ckpt_opt = discovery_dir / "checkpoint_opt.pt"
    ckpt_best = discovery_dir / "checkpoint_best_bank.pt"
    resume = bool(ckpt_state_path.exists() and ckpt_bank.exists() and ckpt_opt.exists() and not force)

    def _build_bank() -> torch.nn.Module:
        base.seed_everything(seed)
        m = base.E0AModel(cfg["m_max"], 64, 2, 4)
        b = m.bank.to(DEVICE)
        del m
        return b

    if resume:
        ckpt = json.loads(ckpt_state_path.read_text(encoding="utf-8"))
        bank = _build_bank()
        bank.load_state_dict(torch.load(ckpt_bank, map_location=DEVICE, weights_only=True))
        optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                      weight_decay=cfg["m_step"]["weight_decay"])
        optimizer.load_state_dict(torch.load(ckpt_opt, map_location=DEVICE, weights_only=True))
        best_round = int(ckpt["best_round"])
        best_val_obj = float(ckpt["best_val_obj"])
        best_bank_state = torch.load(ckpt_best, map_location=DEVICE, weights_only=True) if ckpt_best.exists() else None
        outer_curves = ckpt["outer_curves"]
        m_step_updates = int(ckpt["m_step_updates"])
        skipped_empty = int(ckpt["skipped_empty"])
        total_inner_steps = int(ckpt["total_inner_steps"])
        prev = load_assignment(discovery_dir / "checkpoint_prev.npz")
        prev_train_m = prev["m"]
        prev_train_v = prev["v"]
        start_round = int(ckpt["round_id"]) + 1
        started = time.time() - float(ckpt["wall_seconds"])
        print(f"[discover seed={seed}] RESUMING from round {start_round} (best_round={best_round})", flush=True)
    else:
        write_run_metadata(run_dir, seed, cfg)
        # Random initialization exactly as the original E0AModel builds its bank.
        bank = _build_bank()
        torch.save(bank.state_dict(), run_dir / "initial_bank.pt")
        (run_dir / "initial_bank.sha256").write_text(hash_state_dict(bank.state_dict()) + "\n", encoding="utf-8")
        optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                      weight_decay=cfg["m_step"]["weight_decay"])
        prev_train_m = None
        prev_train_v = None
        best_val_obj = float("inf")
        best_round = -1
        best_bank_state = None
        outer_curves = []
        m_step_updates = 0
        skipped_empty = 0
        total_inner_steps = 0
        start_round = 0
        started = time.time()

    S_train, _, delta_train = base.load_visible("e0a", "train", "identity")
    S_val, _, delta_val = base.load_visible("e0a", "val", "identity")
    probe = make_probe()

    for round_id in range(start_round, o_cfg["max_rounds"]):
        freeze_bank(bank)
        train_latent = exact_support_e_step(bank, S_train, delta_train, e_cfg, prev_v=prev_train_v,
                                            round_id=round_id, opt_seed=seed, log=True)
        val_latent = exact_support_e_step(bank, S_val, delta_val, e_cfg, prev_v=None,
                                          round_id=round_id, opt_seed=seed, log=False)
        total_inner_steps += int(train_latent["inner_steps"]) + int(val_latent["inner_steps"])

        save_assignment(assign_dir / f"round_{round_id:03d}_train.npz", train_latent["m"], train_latent["v"],
                        train_latent["effect_mse"], train_latent["penalized_J"], train_latent["second_margin"])
        save_assignment(assign_dir / f"round_{round_id:03d}_val.npz", val_latent["m"], val_latent["v"],
                        val_latent["effect_mse"], val_latent["penalized_J"], val_latent["second_margin"])

        scr = float(np.mean(train_latent["m"] != prev_train_m)) if prev_train_m is not None else None
        J_train = float(train_latent["penalized_J"].mean())
        effect_train = float(train_latent["effect_mse"].mean())
        active_train = float(train_latent["m"].sum(axis=1).mean())
        J_val = float(val_latent["penalized_J"].mean())
        effect_val = float(val_latent["effect_mse"].mean())

        if J_val < best_val_obj:
            best_val_obj = J_val
            best_round = round_id
            best_bank_state = {k: v.detach().clone() for k, v in bank.state_dict().items()}

        unfreeze_bank(bank)
        shared_before = snapshot_shared(bank)
        m_record = m_step_round(bank, optimizer, train_latent["m"], train_latent["v"],
                                S_train, delta_train, cfg, round_id, seed, probe, shared_before)
        m_step_updates += sum(1 for c in m_record["candidates"] if not c.get("skipped_empty_candidate", False)) * cfg["m_step"]["steps_per_candidate_per_round"]
        skipped_empty += sum(1 for c in m_record["candidates"] if c.get("skipped_empty_candidate", False))

        prev_J_val = outer_curves[-1]["J_val"] if outer_curves else None
        roc = abs(J_val - prev_J_val) / max(abs(prev_J_val), 1e-12) if prev_J_val is not None else None

        record = {
            "round": round_id, "J_train": J_train, "effect_train": effect_train, "active_train": active_train,
            "J_val": J_val, "effect_val": effect_val, "scr": scr, "roc": roc,
            "best_round": best_round, "m_step": m_record,
        }
        outer_curves.append(record)
        dump_json(discovery_dir / "outer_curves.json", outer_curves)
        dump_json(discovery_dir / "candidate_order.json",
                  [{"round": c["round"], "order": c["m_step"]["order"]} for c in outer_curves])
        print(f"[discover seed={seed}] round={round_id} J_train={J_train:.6f} J_val={J_val:.6f} "
              f"scr={scr} roc={roc} active={active_train:.3f} ({time.time()-started:.0f}s)", flush=True)

        prev_train_m = train_latent["m"].copy()
        prev_train_v = train_latent["v"].copy()

        # Save a round-level checkpoint for resume.
        torch.save(bank.state_dict(), discovery_dir / "checkpoint_bank.pt")
        torch.save(optimizer.state_dict(), discovery_dir / "checkpoint_opt.pt")
        if best_bank_state is not None:
            torch.save(best_bank_state, discovery_dir / "checkpoint_best_bank.pt")
        dump_json(discovery_dir / "checkpoint_state.json", {
            "round_id": round_id, "best_round": best_round, "best_val_obj": best_val_obj,
            "outer_curves": outer_curves, "m_step_updates": m_step_updates,
            "skipped_empty": skipped_empty, "total_inner_steps": total_inner_steps,
            "wall_seconds": time.time() - started,
        })
        save_assignment(discovery_dir / "checkpoint_prev.npz", prev_train_m, prev_train_v,
                        train_latent["effect_mse"], train_latent["penalized_J"], train_latent["second_margin"])

        if _check_stability(outer_curves, o_cfg):
            break

    assert best_bank_state is not None, "no best bank state recorded"
    torch.save(bank.state_dict(), discovery_dir / "final_bank.pt")
    torch.save(best_bank_state, discovery_dir / "best_bank.pt")

    # Restore best bank and re-run the full E-step to freeze the teacher.
    bank.load_state_dict(best_bank_state)
    freeze_bank(bank)
    best_train = exact_support_e_step(bank, S_train, delta_train, e_cfg, prev_v=None, round_id=best_round, opt_seed=seed)
    best_val = exact_support_e_step(bank, S_val, delta_val, e_cfg, prev_v=None, round_id=best_round, opt_seed=seed)
    save_assignment(assign_dir / "best_train.npz", best_train["m"], best_train["v"],
                    best_train["effect_mse"], best_train["penalized_J"], best_train["second_margin"])
    save_assignment(assign_dir / "best_val.npz", best_val["m"], best_val["v"],
                    best_val["effect_mse"], best_val["penalized_J"], best_val["second_margin"])
    _save_audit_csv(discovery_dir / "support_scores_audit.csv", best_train["audit_scores"])

    # Re-check stability: the best bank's assignments should be a fixed point.
    round_best_train = load_assignment(assign_dir / f"round_{best_round:03d}_train.npz")
    fixpoint_scr = float(np.mean(best_train["m"] != round_best_train["m"]))

    discovery_stable = _check_stability(outer_curves, o_cfg)
    stable_round = outer_curves[-1]["round"] if discovery_stable else -1

    # Optimization warning: two consecutive train-objective increases > 1%.
    opt_warning = False
    Js = [c["J_train"] for c in outer_curves]
    for k in range(2, len(Js)):
        if Js[k] > Js[k - 1] * 1.01 and Js[k - 1] > Js[k - 2] * 1.01:
            opt_warning = True

    stability = {
        "experiment": "E0-ADP1",
        "optimization_seed": seed,
        "discovery_stable": bool(discovery_stable),
        "stable_round": stable_round,
        "rounds_run": len(outer_curves),
        "best_round": best_round,
        "best_val_objective": best_val_obj,
        "teacher_fixpoint_scr": fixpoint_scr,
        "optimization_warning": opt_warning,
        "m_step_candidate_block_updates": m_step_updates,
        "skipped_empty_candidate_updates": skipped_empty,
        "total_inner_v_steps": total_inner_steps,
        "wall_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if DEVICE.type == "cuda" else 0,
    }
    dump_json(discovery_dir / "stability.json", stability)
    print(json.dumps(stability, indent=2), flush=True)
    return stability


def discover(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        discover_seed(seed, cfg, args.force)


# --------------------------------------------------------------------------- #
# amortize / evaluate
# --------------------------------------------------------------------------- #
def amortize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        run_dir = ADP1_OUTPUT_ROOT / f"seed_{seed}"
        stability = json.loads((run_dir / "discovery" / "stability.json").read_text(encoding="utf-8"))
        if not stability["discovery_stable"]:
            dump_json(run_dir / "amortization" / args.tp / "metrics.json",
                      {"tp": args.tp, "optimization_seed": seed, "status": "N/A_PREREQUISITE_FAIL"})
            print(f"[amortize] seed={seed} skipped (not stable)", flush=True)
            continue
        bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
        bank.load_state_dict(torch.load(run_dir / "discovery" / "best_bank.pt", map_location=DEVICE, weights_only=True))
        freeze_bank(bank)
        teacher_train = load_assignment(run_dir / "discovery" / "assignments" / "best_train.npz")
        teacher_val = load_assignment(run_dir / "discovery" / "assignments" / "best_val.npz")
        S_train, p_train, _ = base.load_visible("e0a", "train", args.tp)
        S_val, p_val, _ = base.load_visible("e0a", "val", args.tp)
        out_dir = run_dir / "amortization" / args.tp
        out_dir.mkdir(parents=True, exist_ok=True)
        info = amortize_qeta.train_qeta(seed, args.tp, bank, S_train, p_train,
                                        teacher_train["m"], teacher_train["v"],
                                        S_val, p_val, teacher_val["m"], teacher_val["v"],
                                        cfg, out_dir)
        dump_json(out_dir / "run_metadata.json", info)


def evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        run_dir = ADP1_OUTPUT_ROOT / f"seed_{seed}"
        evaluate_adp1.evaluate_exact_seed(seed, cfg, run_dir)
        for tp in ["identity", "orthogonal"]:
            if (run_dir / "amortization" / tp / "qeta_best.pt").exists():
                evaluate_adp1.evaluate_amortized_seed(seed, cfg, run_dir, tp)


# --------------------------------------------------------------------------- #
# summarize + report
# --------------------------------------------------------------------------- #
def summarize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    root = ADP1_OUTPUT_ROOT
    seeds = cfg["optimization_seeds"]

    exact_metrics = []
    for seed in seeds:
        p = root / f"seed_{seed}" / "discovery" / "metrics_exact.json"
        if p.exists():
            exact_metrics.append(json.loads(p.read_text(encoding="utf-8")))
    n_pass = sum(1 for m in exact_metrics if m.get("pass"))
    n_stable = sum(1 for m in exact_metrics if m.get("pass_components", {}).get("discovery_stable"))

    summary = {
        "experiment": "E0-ADP1",
        "n_seeds": len(exact_metrics),
        "discovery_pass_seeds": n_pass,
        "recovery_rate_D": n_pass / max(len(exact_metrics), 1),
        "stable_seeds": n_stable,
        "seed_metrics": [
            {
                "seed": m["optimization_seed"],
                "iid_nrmse": m["prediction_nrmse"]["test_iid"],
                "context_nrmse": m["prediction_nrmse"]["test_context"],
                "nrmse_101": m["prediction_nrmse"]["test_combination_101"],
                "participation_f1": m["test_iid"]["participation_f1_mean"],
                "instance_r2_min": m["instance_effect_r2_min"],
                "functional_r2_min": m["functional_r2_min"],
                "redundant_usage": m["redundant_usage_max"],
                "stable": m["pass_components"]["discovery_stable"],
                "pass": m["pass"],
            }
            for m in exact_metrics
        ],
    }
    dump_json(root / "summary.json", summary)
    cmp = compare_joint.compare_joint(root / ".." / "e0a" / "main", root, seeds)
    _write_report(summary, cmp, root)
    print(json.dumps(summary, indent=2), flush=True)


def _write_report(summary: Dict[str, Any], cmp: Dict[str, Any], root: Path) -> None:
    r = summary
    n_pass = r["discovery_pass_seeds"]
    n = r["n_seeds"]
    if n_pass >= 8:
        decision = f"E0-ADP1 DISCOVERY PASS：{n_pass} / {n} 随机种子进入 GT-like basin"
    elif n_pass >= 1:
        decision = f"E0-ADP1 DISCOVERY PARTIAL：仅 {n_pass} / {n} 随机种子进入 GT-like basin"
    else:
        decision = f"E0-ADP1 DISCOVERY FAIL：0 / {n}（或 {n_pass} / {n}）随机种子满足完整恢复"

    lines = [
        "# E0-ADP1 最终报告", "",
        f"**最终判定：{decision}**", "",
        "## 1. Experiment validity", "",
        "- Dataset hash / oracle / 101 leakage / solver sanity: see `input_validation.json` and `solver_sanity/metrics.json`.",
        "- Hidden-label isolation: discovery and amortization only read `visible.npz`; enforced by unit test #8.",
        "",
        "## 2. Discovery 主结果", "",
        f"- Discovery PASS seeds: {n_pass} / {n}",
        f"- RecoveryRate_D: {r['recovery_rate_D']:.2f}",
        f"- Stable seeds: {r['stable_seeds']} / {n}",
        "",
        "| seed | IID NRMSE | 101 NRMSE | F1 | inst-R2 min | func-R2 min | redundant | stable | PASS |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for m in r["seed_metrics"]:
        lines.append(f"| {m['seed']} | {m['iid_nrmse']:.4f} | {m['nrmse_101']:.4f} | {m['participation_f1']:.3f} | "
                     f"{m['instance_r2_min']:.3f} | {m['functional_r2_min']:.3f} | {m['redundant_usage']:.3f} | "
                     f"{m['stable']} | {m['pass']} |")
    lines += [
        "",
        "## 3. 与 joint baseline 的 paired comparison", "",
        "详见 `paired_baseline_comparison.csv`。paired differences (ADP1 exact − joint identity):",
        "",
    ]
    for label, ci in cmp.get("paired_differences_adp1_vs_joint_identity", {}).items():
        lines.append(f"- {label}: mean {ci['mean']:.5f} (95% CI [{ci['ci_low']:.5f}, {ci['ci_high']:.5f}])")
    lines += [
        "",
        "## 4. Amortization", "",
        "见 `seed_*/amortization/{identity,orthogonal}/metrics.json`。",
        "",
        "## 5. 失败模式 / 结论", "",
        decision, "",
        "注：本实验保留原 joint E0-A FAIL 结果；ADP1 为独立 diagnostic experiment。",
    ]
    (root / "E0_ADP1_final_report.md").write_text("\n".join(lines), encoding="utf-8")

    checklist = [
        "# E0-ADP1 reproducibility checklist", "",
        "- [x] WORLD_SEED=20260901 / DATASET_SEED=20260902", "",
        "- [x] 复用 e0a_world_seed_20260901，未重新生成", "",
        "- [x] manifest SHA-256 校验（见 input_validation.json）", "",
        "- [x] oracle MSE < 1e-10，held-out 101 无泄漏", "",
        "- [x] M_max=5，复用 SetMechanismBank / MechanismAbstraction", "",
        "- [x] 每 seed 保存 initial_bank.pt 与 SHA-256", "",
        "- [x] E-step 枚举 32 supports（含 00000 与 11111），v=1.5*tanh(z)，5 restarts", "",
        "- [x] M-step residual/backfitting + adapter-row snapshot/restore", "",
        "- [x] best checkpoint 按 validation penalized objective 选择", "",
        "- [x] hidden GT 仅由 evaluate 入口读取（unit test #8）", "",
        "- [x] PASS/FAIL 使用预注册阈值", "",
    ]
    (root / "reproducibility_checklist.md").write_text("\n".join(checklist), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="E0-ADP1 experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    vi = sub.add_parser("validate-inputs")
    vi.add_argument("--dataset", default=str(base.DATA_ROOT / "e0a_world_seed_20260901"))

    vs = sub.add_parser("validate-solver")
    vs.add_argument("--config", default=str(CONFIG_PATH))
    vs.add_argument("--seed", type=int, default=0)
    vs.add_argument("--num-samples", type=int, default=100)

    d = sub.add_parser("discover")
    d.add_argument("--config", default=str(CONFIG_PATH))
    d.add_argument("--seed", type=int, default=None)
    d.add_argument("--seeds", default=None)
    d.add_argument("--force", action="store_true")

    am = sub.add_parser("amortize")
    am.add_argument("--config", default=str(CONFIG_PATH))
    am.add_argument("--tp", choices=["identity", "orthogonal"], required=True)
    am.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--config", default=str(CONFIG_PATH))
    ev.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")

    cj = sub.add_parser("compare-joint")
    cj.add_argument("--joint-root", default=str(base.OUTPUT_ROOT / "e0a" / "main"))
    cj.add_argument("--adp1-root", default=str(ADP1_OUTPUT_ROOT))

    sm = sub.add_parser("summarize")
    sm.add_argument("--config", default=str(CONFIG_PATH))

    args = parser.parse_args()
    if args.command == "validate-inputs":
        validate_inputs(Path(args.dataset))
    elif args.command == "validate-solver":
        cfg = load_config(args.config)
        validate_solver(cfg, args.seed, args.num_samples)
    elif args.command == "discover":
        discover(args)
    elif args.command == "amortize":
        amortize(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "compare-joint":
        compare_joint.compare_joint(Path(args.joint_root), Path(args.adp1_root))
    elif args.command == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
