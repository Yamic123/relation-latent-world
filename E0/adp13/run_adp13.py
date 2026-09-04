"""ADP13: sparse posterior competition and predictive-evidence selection."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_DLL_HANDLE = None
_DLL_DIR = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and _DLL_DIR.is_dir():
    os.environ["PATH"] = str(_DLL_DIR) + os.pathsep + os.environ.get("PATH", "")
    _DLL_HANDLE = os.add_dll_directory(str(_DLL_DIR))

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_e0 as base
import adp6.run_adp6 as adp6
import adp7.run_adp7 as adp7
import adp9.run_adp9 as adp9
import adp11.run_adp11 as adp11
import adp12.run_adp12 as adp12
from adp1._utils import hash_state_dict

OUT = ROOT / "outputs" / "e0_adp13_sparse_bayesian"
CONFIG = ROOT / "configs" / "e0_adp13.json"
DEVICE = base.DEVICE
VARIANTS = {
    "S1": (2, False, "s1_top2_no_evidence"),
    "S2": (2, True, "s2_top2_evidence"),
    "S3": (3, True, "s3_top3_evidence"),
    "S4": (1, True, "s4_top1_evidence"),
}


def cv(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    raise TypeError(type(x))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=cv), encoding="utf-8")


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, default=cv) + "\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def setup(variant, seed):
    k, evidence, name = VARIANTS[variant]
    c = read(CONFIG)
    c.update({"optimization_seed": seed, "variant": variant, "top_k": k, "evidence_selection": evidence})
    root = OUT / name / f"seed_{seed:03d}"
    cp = root / "audit" / "config.json"
    dump(cp, c)
    adp6.OUT, adp6.CONFIG = root, cp
    adp9.OUT, adp9.A2, adp9.CONFIG = root, root / "training", cp
    adp7.A2 = adp9.A2
    return root, c, k, evidence


def sparse_posterior(cost, alive, top_k, kappa, eps):
    alive = np.asarray(alive, dtype=np.int64)
    z = cost[:, alive]
    ordered = np.sort(z, axis=1)
    gap = float(np.median(ordered[:, 1] - ordered[:, 0])) if len(alive) > 1 else 0.0
    tau = kappa * max(gap, eps)
    logits = -z / tau
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    full = np.zeros_like(cost, dtype=np.float32)
    full[:, alive] = p
    kk = min(top_k, len(alive))
    order = np.argsort(-p, axis=1, kind="stable")[:, :kk]
    membership = np.zeros_like(cost, dtype=bool)
    rows = np.arange(len(cost))[:, None]
    membership[rows, alive[order]] = True
    sparse = np.where(membership, full, 0.0)
    sparse /= sparse.sum(axis=1, keepdims=True)
    return full, sparse.astype(np.float32), membership, gap, tau


def entropy(q):
    return float(np.mean(-(q * np.log(np.maximum(q, 1e-12))).sum(axis=1)))


def evaluate(bank, a, b, tr, va, hv, atr, av, avh, alive, top_k, kappa, c, previous, r, sec, frozen):
    cost = adp6.family_e_step_structured(bank, a, b, tr["S"], tr["response_train"], atr)[1]
    full, q, member, gap, tau = sparse_posterior(cost, alive, top_k, kappa, c["temperature_epsilon"])
    vacost = adp6.family_e_step_structured(bank, a, b, va["S"], va["response_train"], av)[1]
    vfull, vq, vmember, _, _ = sparse_posterior(vacost, alive, top_k, kappa, c["temperature_epsilon"])
    assignment, vassign = q.argmax(1), vq.argmax(1)
    train_pred, _ = adp12.predict_mix(bank, a, b, tr["S"], atr, q)
    held, _ = adp12.predict_mix(bank, a, b, va["S"], avh, vq)
    _, vv = adp6.predict(bank, va["S"], vassign, av, a, b)
    coord = adp6.coordinate_metrics(av, vv, vassign, hv)
    fm = adp6.functional_eval(bank, a, b, av, vassign, hv, c)
    _, _, acc, frag, _ = adp6.fixed_structure(vassign, hv)
    sorted_full = np.sort(full[:, alive], axis=1)
    row = {
        "round": r,
        "top_k": top_k,
        "temperature_kappa": kappa,
        "cost_gap_median": gap,
        "temperature": tau,
        "full_entropy_mean": entropy(full),
        "sparse_entropy_mean": entropy(q),
        "top1_mass_mean": float(sorted_full[:, -1].mean()),
        "top2_mass_mean": float(sorted_full[:, -min(2, len(alive)):].sum(1).mean()),
        "top1_top2_margin_mean": float((sorted_full[:, -1] - sorted_full[:, -2]).mean()) if len(alive) > 1 else 1.0,
        "weighted_responsibility": q.mean(0).tolist(),
        "topk_inclusion_frequency": member.mean(0).tolist(),
        "candidate_usage": (np.bincount(assignment, minlength=5) / len(q)).tolist(),
        "candidate_family_count": np.bincount(assignment, minlength=5).tolist(),
        "family_assignment_change": float(np.mean(assignment != previous)) if previous is not None else 1.0,
        "alive": list(alive),
        "heldout_realization_nrmse": base.nrmse(va["response_heldout"], held),
        "train_nrmse": base.nrmse(tr["response_train"], train_pred),
        "candidate_a": a.detach().cpu().tolist(),
        "candidate_b": b.detach().cpu().tolist(),
        "family_hungarian_accuracy": acc,
        "mean_GT_fragmentation": float(np.mean(frag)),
        "m_step_seconds": sec,
        "mechanisms_frozen": frozen,
        **coord,
        **fm,
    }
    aux = {"cost": cost, "full": full, "q": q, "member": member, "val_cost": vacost, "vfull": vfull, "vq": vq, "vmember": vmember, "held": held}
    return row, aux


def existence_test(j, cost, qkeep, bank, a, b, x, alpha, alpha_h, alive, top_k, kappa, c, r):
    remaining = [z for z in alive if z != j]
    _, qdrop, _, _, _ = sparse_posterior(cost, remaining, top_k, kappa, c["temperature_epsilon"])
    keep_obs, _ = adp12.predict_mix(bank, a, b, x["S"], alpha, qkeep)
    drop_obs, _ = adp12.predict_mix(bank, a, b, x["S"], alpha, qdrop)
    keep_held, _ = adp12.predict_mix(bank, a, b, x["S"], alpha_h, qkeep)
    drop_held, _ = adp12.predict_mix(bank, a, b, x["S"], alpha_h, qdrop)
    regions, bases = adp9.state_regions(x["S"]), adp9.base_regions(x["base_id"])
    domains = {
        "iid": (x["response_train"], keep_obs, drop_obs, np.ones(len(qkeep), bool)),
        "state": (x["response_heldout"], keep_held, drop_held, regions["extreme"]),
        "realization": (x["response_heldout"], keep_held, drop_held, np.ones(len(qkeep), bool)),
        "base": (x["response_heldout"], keep_held, drop_held, bases["base_B"]),
    }
    keep_loss, drop_loss, delta, upper, ci = {}, {}, {}, {}, {}
    rng = np.random.default_rng(c["optimization_seed"] * 100003 + r * 101 + j)
    uids = np.unique(x["base_id"])
    for name, (y, pk, pd, mask) in domains.items():
        keep_loss[name] = adp12.nrmse_mask(y, pk, mask)
        drop_loss[name] = adp12.nrmse_mask(y, pd, mask)
        delta[name] = drop_loss[name] - keep_loss[name]
        values = []
        for _ in range(c["existence_bootstrap_repeats"]):
            ids = rng.choice(uids, len(uids), replace=True)
            idx = np.concatenate([np.flatnonzero((x["base_id"] == uid) & mask) for uid in ids])
            values.append(base.nrmse(y[idx], pd[idx]) - base.nrmse(y[idx], pk[idx]) if len(idx) else 0.0)
        ci[name] = [float(np.quantile(values, .025)), float(np.quantile(values, .975))]
        upper[name] = ci[name][1]
    before_sub = adp12.subgroup_nrmse(x["response_heldout"], keep_held, x, alpha_h)
    after_sub = adp12.subgroup_nrmse(x["response_heldout"], drop_held, x, alpha_h)
    subgroup_delta = [u - v for u, v in zip(after_sub, before_sub)]
    base_delta = []
    for uid in uids:
        mask = x["base_id"] == uid
        base_delta.append(base.nrmse(x["response_heldout"][mask], drop_held[mask]) - base.nrmse(x["response_heldout"][mask], keep_held[mask]))
    finite = bool(np.isfinite(cost[:, remaining]).any(axis=1).all())
    checks = {
        "bootstrap_all_domains": max(upper.values()) < c["existence_global_delta_nrmse"],
        "subgroup": max(subgroup_delta) < c["existence_subgroup_delta_nrmse"],
        "finite_remaining": finite,
        "global_heldout": delta["realization"] < c["existence_global_delta_nrmse"],
    }
    return {
        "candidate": f"C{j+1}", "candidate_index": j, "round": r,
        "keep_losses": keep_loss, "drop_losses": drop_loss, "domain_delta_nrmse": delta,
        "bootstrap_ci95": ci, "bootstrap_upper_95": upper,
        "base_delta_mean": float(np.mean(base_delta)), "base_delta_median": float(np.median(base_delta)),
        "base_delta_p95": float(np.quantile(base_delta, .95)), "base_level_delta": base_delta,
        "max_subgroup_delta_nrmse": float(max(subgroup_delta)), "post_delete_subgroup_nrmse": after_sub,
        "score": float(max(upper.values())), "checks": checks, "pass": all(checks.values()),
        "post_delete_heldout_nrmse": drop_loss["realization"],
    }


def train(variant, seed, root, c, top_k, evidence, force=False):
    final = root / "training" / "final_metrics.json"
    if final.exists() and not force: return read(final)
    adp11.prepare_coordinates(root)
    tr, va, hv = adp6.response_data("train"), adp6.response_data("val"), adp6.evaluator_data("val")
    atr, _ = adp6.coordinates("train")
    av, avh = adp6.coordinates("val")
    bank, a, b, q = adp12.init(seed, c, len(tr["S"]))
    alive, previous = list(range(5)), q.argmax(1)
    opt = torch.optim.AdamW([
        {"params": bank.parameters(), "lr": c["mechanism_learning_rate"], "weight_decay": c["weight_decay"]},
        {"params": [a, b], "lr": c["coordinate_learning_rate"], "weight_decay": 0.0},
    ])
    for path in (root / "training/checkpoints", root / "training/assignments", root / "posterior", root / "existence_tests"):
        path.mkdir(parents=True, exist_ok=True)
    dump(root / "audit/initial.json", {"initial_model_hash": hash_state_dict(bank.state_dict()), "initial_hard_count": np.bincount(previous, minlength=5), "no_checkpoint_loaded": True})
    rows, events, pending, started = [], [], None, time.time()
    for r in range(c["rounds"]):
        frozen = pending is not None
        if frozen: mse, sec = float("nan"), 0.0
        else: mse, sec = adp12.weighted_train(bank, a, b, tr["S"], tr["response_train"], atr, q, c, opt, r)
        kappa = c["temperature_kappa"][min(r // 10, 3)]
        row, aux = evaluate(bank, a, b, tr, va, hv, atr, av, avh, alive, top_k, kappa, c, previous, r, sec, frozen)
        q, previous = aux["q"], aux["q"].argmax(1)
        event_summary = None
        if pending is not None:
            current_sub = adp12.subgroup_nrmse(va["response_heldout"], aux["held"], va, avh)
            global_drift = row["heldout_realization_nrmse"] - pending["expected_nrmse"]
            subgroup_drift = max(x-y for x, y in zip(current_sub, pending["expected_subgroup"]))
            func = np.asarray(row["functional_r2_matrix"])[alive]
            functional_ok = bool(np.all(np.max(func, axis=0) >= c["rollback_min_functional_r2"]))
            rollback = global_drift > c["existence_global_delta_nrmse"] or subgroup_drift > c["existence_subgroup_delta_nrmse"] or not functional_ok
            pending["event"].update({"validation_round": r, "frozen_global_drift": global_drift, "frozen_max_subgroup_drift": subgroup_drift, "frozen_functional_ok": functional_ok, "rollback": rollback})
            if rollback:
                alive.append(pending["j"]); alive.sort()
                _, q, _, _, _ = sparse_posterior(aux["cost"], alive, top_k, kappa, c["temperature_epsilon"])
                previous = q.argmax(1)
            dump(pending["path"], pending["event"])
            event_summary = {"validation_of": pending["event"]["candidate"], "rollback": rollback}
            pending = None
        if evidence and pending is None and r in c["existence_rounds"] and len(alive) > 1:
            tests = [existence_test(j, aux["val_cost"], aux["vq"], bank, a, b, va, av, avh, alive, top_k, kappa, c, r) for j in alive]
            eligible = [x for x in tests if x["pass"]]
            chosen = min(eligible, key=lambda x: x["score"]) if eligible else None
            for test in tests:
                test["decision"] = "delete" if test is chosen else ("retain_higher_score" if test["pass"] else "retain_gate_fail")
                path = root / "existence_tests" / f"round_{r:03d}_{test['candidate']}.json"
                dump(path, test)
            if chosen is not None:
                j = chosen["candidate_index"]
                alive.remove(j)
                chosen["commit"] = True
                chosen_path = root / "existence_tests" / f"round_{r:03d}_{chosen['candidate']}.json"
                dump(chosen_path, chosen)
                events.append(chosen)
                _, q, _, _, _ = sparse_posterior(aux["cost"], alive, top_k, kappa, c["temperature_epsilon"])
                previous = q.argmax(1)
                if r < c["rounds"] - 1:
                    pending = {"j": j, "event": chosen, "path": chosen_path, "expected_nrmse": chosen["post_delete_heldout_nrmse"], "expected_subgroup": chosen["post_delete_subgroup_nrmse"]}
                else:
                    func = np.asarray(row["functional_r2_matrix"])[alive]
                    functional_ok = bool(np.all(np.max(func, axis=0) >= c["rollback_min_functional_r2"]))
                    rollback = not functional_ok
                    chosen.update({"validation_round": "final_immediate", "frozen_global_drift": 0.0, "frozen_max_subgroup_drift": 0.0, "frozen_functional_ok": functional_ok, "rollback": rollback})
                    if rollback:
                        alive.append(j); alive.sort()
                    dump(chosen_path, chosen)
                event_summary = {"tested": len(tests), "selected": chosen["candidate"], "score": chosen["score"]}
            else:
                event_summary = {"tested": len(tests), "selected": None}
        row["existence_event"] = event_summary
        row["alive_after_selection"] = alive.copy()
        rows.append(row)
        # Save post-selection assignment so ADP9 never receives a deleted label.
        if set(np.unique(previous)) - set(alive):
            _, q, _, _, _ = sparse_posterior(aux["cost"], alive, top_k, kappa, c["temperature_epsilon"])
            previous = q.argmax(1)
        _, vq_save, _, _, _ = sparse_posterior(aux["val_cost"], alive, top_k, kappa, c["temperature_epsilon"])
        append(root / "training/round_metrics.jsonl", row)
        np.savez_compressed(root / "posterior" / f"round_{r:03d}.npz", full_q=aux["full"], sparse_q=aux["q"], topk=aux["member"], alive_pre=np.asarray(row["alive"]), alive_post=np.asarray(alive), cost=aux["cost"])
        np.savez_compressed(root / "training/assignments" / f"round_{r:03d}.npz", train=previous, val=vq_save.argmax(1), train_cost=aux["cost"], val_cost=aux["val_cost"])
        torch.save({"bank": bank.state_dict(), "coordinate_a": a.detach().cpu(), "coordinate_b": b.detach().cpu()}, root / "training/checkpoints" / f"round_{r:03d}.pt")
        print(f"[{variant} s{seed}] r={r:02d} Hfull={row['full_entropy_mean']:.3f} Hsparse={row['sparse_entropy_mean']:.3f} alive={alive} incl={np.round(row['topk_inclusion_frequency'],3).tolist()} nrmse={row['heldout_realization_nrmse']:.3f} func={np.round(row['matched_functional_r2'],2).tolist()}", flush=True)
    # Re-evaluate after a possible final-round deletion, without an extra training update.
    if rows[-1]["alive"] != alive:
        last, _ = evaluate(bank, a, b, tr, va, hv, atr, av, avh, alive, top_k, c["temperature_kappa"][-1], c, previous, 40, 0.0, True)
    else:
        last = rows[-1]
    checks = adp12.z4_gate(last, c, alive)
    result = {
        "stage": "adp13_training", "variant": variant, "seed": seed, "top_k": top_k, "evidence": evidence,
        "alive": alive, "existence_events": events,
        "existence_deletes": sum(x.get("commit", False) and not x.get("rollback", False) for x in events),
        "existence_rollbacks": sum(x.get("rollback", False) for x in events),
        "last_round": last, "z4_checks": checks, "z4_pass": all(checks.values()), "wall_seconds": time.time() - started,
    }
    dump(final, result)
    return result


def failure(training, pruning):
    if not training["z4_pass"]: return "F5_persistent_mixing_or_identity_failure"
    if not pruning or pruning["M_discovered"] > 3: return "F7_count_only_failure"
    if pruning["M_discovered"] < 3: return "F3_evidence_false_death"
    if not pruning["pass"]: return "F6_duplicate_or_final_gate_failure"
    return None


def run_one(variant, seed, force=False):
    root, c, top_k, evidence = setup(variant, seed)
    final = root / "final_verdict.json"
    if final.exists() and not force: return read(final)
    training = train(variant, seed, root, c, top_k, evidence, force)
    pruning = adp12.duplicate_prune(root, c, training) if training["z4_pass"] else None
    mode = failure(training, pruning)
    b1 = adp12.run_b1(root, c, pruning) if mode is None else None
    if mode is None and not b1["pass"]: mode = "F8_B1_failure"
    result = {
        "variant": variant, "seed": seed, "top_k": top_k, "z4_pass": training["z4_pass"],
        "major_mixing": not training["z4_checks"].get("purity", False),
        "existence_tests": len(list((root / "existence_tests").glob("round_*.json"))),
        "existence_deletes": training["existence_deletes"], "existence_rollbacks": training["existence_rollbacks"],
        "duplicate_prunes": sum(x["commit"] for x in pruning["telemetry"]) if pruning else 0,
        "m_discovered": pruning["M_discovered"] if pruning else None,
        "family_accuracy": pruning["metrics"]["family_hungarian_accuracy"] if pruning else training["last_round"]["family_hungarian_accuracy"],
        "fragmentation": pruning["metrics"]["mean_fragmentation"] if pruning else training["last_round"]["mean_GT_fragmentation"],
        "b1_pass": bool(b1 and b1["pass"]), "final_pass": mode is None, "failure_mode": mode,
    }
    dump(final, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def baselines():
    rows = []
    sources = {
        "H0": ROOT / "outputs/e0_adp11_from_zero/identity_tp",
        "H2": ROOT / "outputs/e0_adp12_bayesian_responsibility/h2_hard_evidence_death",
        "H3": ROOT / "outputs/e0_adp12_bayesian_responsibility/h3_soft_evidence_death",
    }
    for variant, source in sources.items():
        for seed in range(3):
            x = read(source / f"seed_{seed:03d}" / "final_verdict.json")
            rows.append({"variant": variant, "seed": seed, "z4_pass": x["z4_pass"], "final_pass": x["final_pass"], "m_discovered": x.get("m_discovered"), "failure_mode": x.get("failure_mode")})
    return rows


def aggregate(seeds):
    new = [read(OUT / name / f"seed_{s:03d}" / "final_verdict.json") for _, _, name in VARIANTS.values() for s in seeds]
    rows = baselines() + new
    target = OUT / "aggregate_3seed"
    target.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "seed", "top_k", "z4_pass", "major_mixing", "existence_tests", "existence_deletes", "existence_rollbacks", "duplicate_prunes", "m_discovered", "family_accuracy", "fragmentation", "b1_pass", "final_pass", "failure_mode"]
    with (target / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    by = {v: {"z4_pass": sum(x["z4_pass"] for x in new if x["variant"] == v), "major_mixing": sum(x["major_mixing"] for x in new if x["variant"] == v), "end_to_end_pass": sum(x["final_pass"] for x in new if x["variant"] == v), "rollbacks": sum(x["existence_rollbacks"] for x in new if x["variant"] == v)} for v in VARIANTS}
    gate = by["S2"]["end_to_end_pass"] >= 2 and by["S2"]["z4_pass"] >= 2 and by["S2"]["major_mixing"] <= 1 and by["S2"]["rollbacks"] == 0
    result = {"stage": "adp13_pilot", "seeds": seeds, "variants": by, "s2_pilot_gate": gate, "pass": gate, "decision": "PROCEED_STRONG" if gate else "STOP"}
    dump(target / "final_verdict.json", result)
    return result


def smoke():
    cost = np.asarray([[1., 2., 3., 4., 5.], [5., 1., 2., 3., 4.]], np.float32)
    for k in (1, 2, 3):
        full, sparse, member, _, _ = sparse_posterior(cost, list(range(5)), k, 2.0, 1e-8)
        assert np.allclose(full.sum(1), 1) and np.allclose(sparse.sum(1), 1)
        assert np.all(member.sum(1) == k) and np.all((sparse > 0).sum(1) == k)
    # Reuse the full CUDA/data-path ADP12 smoke test as the lower-level integration check.
    import adp12.smoke_adp12 as lower
    lower.main()
    root = OUT / "_smoke" / "one_round_integration"
    c = read(CONFIG)
    c.update({"optimization_seed": 998, "variant": "S2", "top_k": 2, "evidence_selection": True,
              "rounds": 1, "epochs_per_round": 1, "family_batch_size": 64,
              "functional_probe_count": 64, "existence_rounds": [0], "existence_bootstrap_repeats": 3})
    cp = root / "audit/config.json"; dump(cp, c)
    adp6.OUT, adp6.CONFIG = root, cp
    adp9.OUT, adp9.A2, adp9.CONFIG = root, root / "training", cp
    adp7.A2 = adp9.A2
    integration = train("S2", 998, root, c, 2, True, force=True)
    result = {"pass": True, "device": str(DEVICE), "top_k_checked": [1, 2, 3],
              "integration_rounds": 1, "existence_files": len(list((root / "existence_tests").glob("*.json"))),
              "integration_alive": integration["alive"]}
    dump(OUT / "_smoke/final_metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=["run", "smoke", "pilot_all", "aggregate", "strong_test"], default="run")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.stage == "smoke": result = smoke()
    elif args.stage == "run":
        if args.variant is None or args.seed is None: parser.error("--variant and --seed are required for stage run")
        result = run_one(args.variant, args.seed, args.force)
    elif args.stage == "pilot_all":
        for variant in VARIANTS:
            for seed in seeds: run_one(variant, seed, args.force)
        result = aggregate(seeds)
    elif args.stage == "strong_test":
        for seed in seeds: run_one("S2", seed, args.force)
        result = aggregate(seeds)
    else: result = aggregate(seeds)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
