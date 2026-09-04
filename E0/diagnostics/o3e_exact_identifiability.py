from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from common import (
    DEVICE,
    OUTPUT_ROOT,
    RunLogger,
    base,
    dump_config,
    dump_json,
    load_all,
    metadata,
    seed_record,
)


PATTERNS = np.asarray(list(itertools.product([0, 1], repeat=3)), dtype=np.float32)


def pattern_name(pattern: Sequence[float]) -> str:
    return "".join(str(int(value)) for value in pattern)


def make_cases(support: str, restarts: int, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
    masks: List[np.ndarray] = []
    lows: List[np.ndarray] = []
    highs: List[np.ndarray] = []
    pattern_ids: List[int] = []
    restart_ids: List[int] = []
    sign_ids: List[str] = []
    initial: List[np.ndarray] = []
    fractions = np.linspace(0.08, 0.92, restarts, dtype=np.float32)

    for pattern_id, mask in enumerate(PATTERNS):
        active = np.flatnonzero(mask)
        assignments = list(itertools.product([-1, 1], repeat=len(active))) if support == "matched" else [()]
        for signs in assignments:
            low = np.zeros(3, dtype=np.float32)
            high = np.zeros(3, dtype=np.float32)
            if support == "matched":
                for index, sign in zip(active, signs):
                    low[index], high[index] = ((-1.0, -0.3) if sign < 0 else (0.3, 1.0))
            else:
                low[active], high[active] = -1.5, 1.5
            for restart in range(restarts):
                value = np.zeros(3, dtype=np.float32)
                if len(active):
                    # Stratified starts plus independent jitter avoid a diagonal-only multivariate search.
                    frac = np.clip(
                        fractions[restart] + rng.uniform(-0.07, 0.07, len(active)), 0.02, 0.98
                    )
                    value[active] = low[active] + frac * (high[active] - low[active])
                masks.append(mask.copy())
                lows.append(low)
                highs.append(high)
                pattern_ids.append(pattern_id)
                restart_ids.append(restart)
                sign_ids.append("".join("+" if sign > 0 else "-" for sign in signs))
                initial.append(value)
    return (
        np.stack(masks), np.stack(lows), np.stack(highs), np.asarray(pattern_ids),
        np.asarray(restart_ids), np.asarray(sign_ids), np.stack(initial),
    )


def effect_basis(S: np.ndarray, world: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    linear = np.einsum("jkab,nkb->njka", world["A"], S, optimize=True)
    support = world["C"][None, :, :, None]
    first = support * (linear + world["b"][None])
    quadratic = np.broadcast_to(
        0.25 * support * world["c"][None], (len(S), 3, 3, 2)
    ).copy()
    return first.reshape(len(S), 3, 6), quadratic.reshape(len(S), 3, 6)


def batched_multistart(
    first: np.ndarray,
    quadratic: np.ndarray,
    delta: np.ndarray,
    masks: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    initial: np.ndarray,
    steps: int,
    learning_rate: float,
    logger: RunLogger,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    n, q = len(first), len(masks)
    first_t = torch.from_numpy(first).to(DEVICE)
    quadratic_t = torch.from_numpy(quadratic).to(DEVICE)
    delta_t = torch.from_numpy(delta.reshape(n, 6)).to(DEVICE)
    mask_t = torch.from_numpy(masks).to(DEVICE)
    low_t = torch.from_numpy(lows).to(DEVICE)
    high_t = torch.from_numpy(highs).to(DEVICE)
    values = torch.nn.Parameter(torch.from_numpy(initial)[None].repeat(n, 1, 1).to(DEVICE))
    # Break exact cross-sample symmetry while respecting each case's box.
    with torch.no_grad():
        jitter = (torch.rand_like(values) - 0.5) * 0.04 * (high_t - low_t)[None]
        values.add_(jitter)
        values.copy_(torch.maximum(torch.minimum(values, high_t[None]), low_t[None]) * mask_t[None])
    optimizer = torch.optim.Adam([values], lr=learning_rate)
    trace: List[Dict[str, float]] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = torch.einsum("nqj,njd->nqd", values, first_t)
        prediction = prediction + torch.einsum("nqj,njd->nqd", values.square(), quadratic_t)
        mse = (prediction - delta_t[:, None]).square().mean(dim=-1)
        mse.sum().backward()
        optimizer.step()
        with torch.no_grad():
            values.copy_(torch.maximum(torch.minimum(values, high_t[None]), low_t[None]) * mask_t[None])
        if step % 50 == 0 or step == steps - 1:
            entry = {
                "step": step,
                "mean_best_reconstruction_mse": float(mse.min(dim=1).values.mean().item()),
                "max_best_reconstruction_mse": float(mse.min(dim=1).values.max().item()),
            }
            trace.append(entry)
            logger.log(
                f"step={step:04d} mean_best_mse={entry['mean_best_reconstruction_mse']:.8g} "
                f"max_best_mse={entry['max_best_reconstruction_mse']:.8g}"
            )
    return values.detach().cpu().numpy(), trace


def refine_one(
    start: np.ndarray,
    mask: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    first: np.ndarray,
    quadratic: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float, bool]:
    active = np.flatnonzero(mask)
    if len(active) == 0:
        return np.zeros(3, dtype=np.float64), float(np.mean(target**2)), True

    def fun(active_values: np.ndarray) -> Tuple[float, np.ndarray]:
        prediction = np.sum(
            active_values[:, None] * first[active]
            + active_values[:, None] ** 2 * quadratic[active], axis=0
        )
        residual = prediction - target
        objective = float(np.mean(residual**2))
        jacobian = first[active] + 2.0 * active_values[:, None] * quadratic[active]
        gradient = (2.0 / target.size) * np.sum(jacobian * residual[None], axis=1)
        return objective, gradient.astype(np.float64)

    result = minimize(
        fun,
        start[active].astype(np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(float(low[j]), float(high[j])) for j in active],
        options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-10, "maxls": 40},
    )
    value = np.zeros(3, dtype=np.float64)
    value[active] = result.x
    return value, float(result.fun), bool(result.success)


def run(support: str, num_samples: int, restarts: int, steps: int) -> Dict[str, object]:
    output = OUTPUT_ROOT / "O3E" / support
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output / "stdout.log")
    config = {
        "experiment": "O3-E",
        "support": support,
        "split": "test_iid",
        "num_samples": num_samples,
        "patterns": [pattern_name(p) for p in PATTERNS],
        "lambda_participation": 1e-3,
        "multiple_v_initializations": restarts,
        "batched_optimizer": "Adam",
        "batched_steps": steps,
        "batched_learning_rate": 0.05,
        "final_optimizer": "scipy L-BFGS-B",
        "matched_active_support": "[-1,-0.3] union [0.3,1]",
        "learner_active_support": "[-1.5,1.5] (actual 1.5*tanh range)",
        "margin_approximately_zero_tolerance": 1e-6,
    }
    dump_config(output / "config.yaml", config)
    dump_json(output / "seed.json", seed_record(0))
    dump_json(output / "run_metadata.json", metadata("O3-E", 0, config))
    logger.log(f"O3-E support={support} samples={num_samples} device={DEVICE}")
    started = time.time()

    data = load_all("test_iid", "identity")
    S = data["S"][:num_samples]
    delta = data["delta_S"][:num_samples].reshape(num_samples, 6).astype(np.float64)
    gt_m = data["m_gt"][:num_samples]
    gt_v = data["v_gt"][:num_samples]
    world = base.load_world("e0a")
    first, quadratic = effect_basis(S, world)
    first64, quadratic64 = first.astype(np.float64), quadratic.astype(np.float64)

    rng = np.random.default_rng(20260903 if support == "matched" else 20260904)
    masks, lows, highs, pattern_ids, restart_ids, sign_ids, initial = make_cases(support, restarts, rng)
    logger.log(f"search_cases_per_sample={len(masks)} (includes signs and {restarts} restarts)")
    values, trace = batched_multistart(
        first, quadratic, delta.astype(np.float32), masks, lows, highs, initial,
        steps, config["batched_learning_rate"], logger,
    )
    dump_json(output / "training_curves.json", {"optimization_trace": trace})

    objectives = np.empty((num_samples, 8), dtype=np.float64)
    optimized_v = np.zeros((num_samples, 8, 3), dtype=np.float64)
    success = np.zeros((num_samples, 8), dtype=bool)
    selected_case = np.zeros((num_samples, 8), dtype=np.int64)
    for sample in range(num_samples):
        if sample % 100 == 0:
            logger.log(f"L-BFGS-B refinement sample={sample}/{num_samples}")
        for pattern_id in range(8):
            cases = np.flatnonzero(pattern_ids == pattern_id)
            candidate_values = values[sample, cases]
            prediction = (
                np.einsum("qj,jd->qd", candidate_values, first64[sample])
                + np.einsum("qj,jd->qd", candidate_values**2, quadratic64[sample])
            )
            candidate_mse = np.mean((prediction - delta[sample]) ** 2, axis=1)
            case = int(cases[int(np.argmin(candidate_mse))])
            selected_case[sample, pattern_id] = case
            value, mse, ok = refine_one(
                values[sample, case], masks[case], lows[case], highs[case],
                first64[sample], quadratic64[sample], delta[sample],
            )
            optimized_v[sample, pattern_id] = value
            objectives[sample, pattern_id] = mse + 1e-3 * float(PATTERNS[pattern_id].sum())
            success[sample, pattern_id] = ok

    gt_ids = np.asarray([int("".join(str(int(x)) for x in row), 2) for row in gt_m])
    best_ids = objectives.argmin(axis=1)
    gt_objective = objectives[np.arange(num_samples), gt_ids]
    alternatives = objectives.copy()
    alternatives[np.arange(num_samples), gt_ids] = np.inf
    best_alternative = alternatives.min(axis=1)
    margins = best_alternative - gt_objective
    tolerance = float(config["margin_approximately_zero_tolerance"])
    correct = best_ids == gt_ids
    active_errors: List[float] = []
    for i in range(num_samples):
        active = gt_m[i].astype(bool)
        if np.any(active):
            active_errors.extend(np.abs(optimized_v[i, best_ids[i], active] - gt_v[i, active]).tolist())

    np.save(output / "margin_distribution.npy", margins)
    table_path = output / "samplewise_objective_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample", "gt_pattern", "selected_pattern", "correct", "gt_objective",
                  "best_alternative_objective", "margin"] + [f"J_{pattern_name(p)}" for p in PATTERNS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(num_samples):
            row = {
                "sample": i, "gt_pattern": pattern_name(gt_m[i]),
                "selected_pattern": pattern_name(PATTERNS[best_ids[i]]), "correct": int(correct[i]),
                "gt_objective": gt_objective[i], "best_alternative_objective": best_alternative[i],
                "margin": margins[i],
            }
            row.update({f"J_{pattern_name(PATTERNS[j])}": objectives[i, j] for j in range(8)})
            writer.writerow(row)

    recovery_path = output / "pattern_recovery.csv"
    with recovery_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["gt_pattern", "n", "top1_recovery", "median_margin", "positive_margin_fraction",
                  "approximately_zero_fraction", "alternative_beats_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pattern_id, pattern in enumerate(PATTERNS):
            chosen = gt_ids == pattern_id
            if not np.any(chosen):
                continue
            local = margins[chosen]
            writer.writerow({
                "gt_pattern": pattern_name(pattern), "n": int(chosen.sum()),
                "top1_recovery": float(correct[chosen].mean()), "median_margin": float(np.median(local)),
                "positive_margin_fraction": float((local > tolerance).mean()),
                "approximately_zero_fraction": float((np.abs(local) <= tolerance).mean()),
                "alternative_beats_fraction": float((local < -tolerance).mean()),
            })

    metrics: Dict[str, object] = {
        "experiment": "O3-E", "support": support, "num_samples": num_samples,
        "gt_participation_top1_recovery": float(correct.mean()),
        "median_objective_margin": float(np.median(margins)),
        "positive_margin_fraction": float((margins > tolerance).mean()),
        "approximately_zero_margin_fraction": float((np.abs(margins) <= tolerance).mean()),
        "alternative_beats_gt_fraction": float((margins < -tolerance).mean()),
        "active_v_mae_selected_pattern": float(np.mean(active_errors)),
        "lbfgsb_success_fraction": float(success.mean()),
        "pass_components": {
            "top1_recovery_gt_0.95": bool(correct.mean() > 0.95),
            "positive_margin_gt_0.95": bool((margins > tolerance).mean() > 0.95),
        },
        "wall_seconds": time.time() - started,
    }
    metrics["pass"] = bool(all(metrics["pass_components"].values()))
    dump_json(output / "metrics.json", metrics)
    logger.log(str(metrics))
    logger.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", choices=["matched", "learner", "both"], default="both")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    supports = ["matched", "learner"] if args.support == "both" else [args.support]
    results = [run(item, args.num_samples, args.restarts, args.steps) for item in supports]
    if len(results) == 2:
        summary = {"experiment": "O3-E", "matched": results[0], "learner": results[1]}
        dump_json(OUTPUT_ROOT / "O3E" / "summary.json", summary)


if __name__ == "__main__":
    main()
