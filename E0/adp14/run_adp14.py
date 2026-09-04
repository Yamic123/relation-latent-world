"""ADP14: constrained specialist absorption from ADP13-S4 checkpoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
import adp12.run_adp12 as adp12

DEVICE = base.DEVICE
CONFIG = ROOT / "configs" / "e0_adp14.json"
OUT = ROOT / "outputs" / "e0_adp14_specialist_absorption"
SOURCE = ROOT / "outputs" / "e0_adp13_sparse_bayesian" / "s4_top1_evidence"


def cv(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=cv), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def configure_source(seed):
    c = read(CONFIG)
    c["optimization_seed"] = seed
    src = SOURCE / f"seed_{seed:03d}"
    coord_cfg = src / "audit" / "config.json"
    adp6.OUT, adp6.CONFIG = src, coord_cfg
    return c, src


def load_source(seed):
    c, src = configure_source(seed)
    cp_path = src / "training" / "checkpoints" / "round_039.pt"
    asn_path = src / "training" / "assignments" / "round_039.npz"
    cp = torch.load(cp_path, map_location=DEVICE, weights_only=True)
    bank = base.SetMechanismBank(5, c["mechanism_dim"], c["mechanism_heads"]).to(DEVICE)
    bank.load_state_dict(cp["bank"])
    a = torch.nn.Parameter(cp["coordinate_a"].to(DEVICE).clone())
    b = torch.nn.Parameter(cp["coordinate_b"].to(DEVICE).clone())
    with np.load(asn_path) as z:
        saved = {k: z[k] for k in z.files}
    training = read(src / "training" / "final_metrics.json")
    alive = [int(x) for x in training["alive"]]
    tr, va = adp6.response_data("train"), adp6.response_data("val")
    hv = adp6.evaluator_data("val")
    atr, _ = adp6.coordinates("train")
    av, avh = adp6.coordinates("val")
    data = {"tr": tr, "va": va, "hv": hv, "atr": atr, "av": av, "avh": avh}
    audit = {
        "source_checkpoint": str(cp_path), "source_checkpoint_sha256": sha256(cp_path),
        "source_assignment": str(asn_path), "source_assignment_sha256": sha256(asn_path),
        "source_retrained": False, "alive_indices": alive,
        "coordinate_a": a.detach().cpu(), "coordinate_b": b.detach().cpu(),
        "architecture_freeze_interpretation": (
            "SetMechanismBank has a shared trunk. To keep every non-parent mechanism exactly frozen, "
            "ADP14 freezes the shared trunk and updates only the selected parent adapter row and its a,b."
        ),
    }
    return c, src, bank, a, b, saved, alive, data, audit


def clone_model(bank, a, b):
    new = base.SetMechanismBank(5, bank.dim, bank.heads).to(DEVICE)
    new.load_state_dict(copy.deepcopy(bank.state_dict()))
    return new, torch.nn.Parameter(a.detach().clone()), torch.nn.Parameter(b.detach().clone())


def assignments(bank, a, b, alive, data):
    ta, tc = adp9.reassign(alive, bank, a, b, data["tr"], data["atr"])
    va, vc = adp9.reassign(alive, bank, a, b, data["va"], data["av"])
    return ta, tc, va, vc


def gap_stats(cost, assignment, child, parent):
    take = assignment == child
    if not np.any(take):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None,
                "fraction_lt_0p01": None, "fraction_lt_0p02": None}
    g = cost[take, parent] - cost[take, child]
    return {"count": int(take.sum()), "mean": float(g.mean()), "median": float(np.median(g)),
            "p90": float(np.quantile(g, .90)), "max": float(g.max()),
            "fraction_lt_0p01": float(np.mean(g < .01)), "fraction_lt_0p02": float(np.mean(g < .02))}


def pair_scan(alive, train_a, train_c, val_a, val_c, c):
    rows = []
    for child in alive:
        tmask, vmask = train_a == child, val_a == child
        for parent in alive:
            if parent == child: continue
            tother = [x for x in alive if x != child]
            tbest = np.asarray(tother)[train_c[tmask][:, tother].argmin(1)] if tmask.any() else np.array([], int)
            vbest = np.asarray(tother)[val_c[vmask][:, tother].argmin(1)] if vmask.any() else np.array([], int)
            pt = float(np.mean(tbest == parent)) if len(tbest) else 0.0
            pv = float(np.mean(vbest == parent)) if len(vbest) else 0.0
            ts, vs = gap_stats(train_c, train_a, child, parent), gap_stats(val_c, val_a, child, parent)
            low_val = int(vmask.sum()) < c["low_val_family_count"]
            gate = pt >= c["proposal_p_train"] and (low_val or pv >= c["proposal_p_val"])
            rows.append({"child": child, "parent": parent, "pair": f"C{child+1}->C{parent+1}",
                         "P_train": pt, "P_val": pv, "low_val_support": low_val,
                         "train_gap": ts, "val_gap": vs, "proposal_gate": bool(gate)})
    med = {(r["child"], r["parent"]): r["val_gap"]["median"] for r in rows}
    for r in rows:
        fwd = r["val_gap"]["median"]
        rev = med.get((r["parent"], r["child"]))
        r["directionality"] = None if fwd is None or rev is None else float(rev - fwd)
    def key(r):
        vm = r["val_gap"]["median"]
        return (-r["P_val"], -r["P_train"], float("inf") if vm is None else vm,
                -(r["directionality"] if r["directionality"] is not None else -1e30))
    rows.sort(key=key)
    return rows, [r for r in rows if r["proposal_gate"]]


def hard_predictions(bank, a, b, assignment, x, alpha):
    allp = adp7.all_candidate_prediction(bank, x["S"], alpha, a, b)
    return allp[np.arange(len(assignment)), :, assignment]


def domain_payload(bank, a, b, alive, val_assignment, data):
    x, alpha, ah = data["va"], data["av"], data["avh"]
    pi = hard_predictions(bank, a, b, val_assignment, x, alpha)
    ph = hard_predictions(bank, a, b, val_assignment, x, ah)
    sr, br = adp9.state_regions(x["S"]), adp9.base_regions(x["base_id"])
    return {
        "iid": (x["response_train"], pi, np.ones(len(val_assignment), bool)),
        "state": (x["response_heldout"], ph, sr["extreme"]),
        "realization": (x["response_heldout"], ph, np.ones(len(val_assignment), bool)),
        "base": (x["response_heldout"], ph, br["base_B"]),
    }, ph


def bootstrap_compare(keep, absorb, base_ids, repeats, seed):
    rng, ids = np.random.default_rng(seed), np.unique(base_ids)
    result = {}
    for name in keep:
        y, pk, mask = keep[name]; _, pa, _ = absorb[name]
        delta = base.nrmse(y[mask], pa[mask]) - base.nrmse(y[mask], pk[mask])
        vals = []
        eligible = [u for u in ids if np.any((base_ids == u) & mask)]
        for _ in range(repeats):
            picked = rng.choice(eligible, len(eligible), replace=True)
            idx = np.concatenate([np.flatnonzero((base_ids == u) & mask) for u in picked])
            vals.append(base.nrmse(y[idx], pa[idx]) - base.nrmse(y[idx], pk[idx]))
        result[name] = {"delta": float(delta), "bootstrap_mean": float(np.mean(vals)),
                        "bootstrap_median": float(np.median(vals)),
                        "ci95": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))],
                        "upper95": float(np.quantile(vals, .975))}
    return result


def compare_models(pre_bank, pre_a, pre_b, pre_alive, pre_val, bank, a, b, alive, val_a,
                   child, parent, data, c, seed):
    keep, keep_h = domain_payload(pre_bank, pre_a, pre_b, pre_alive, pre_val, data)
    post, post_h = domain_payload(bank, a, b, alive, val_a, data)
    boot = bootstrap_compare(keep, post, data["va"]["base_id"], c["bootstrap_repeats"], seed)
    before_sub = adp12.subgroup_nrmse(data["va"]["response_heldout"], keep_h, data["va"], data["avh"])
    after_sub = adp12.subgroup_nrmse(data["va"]["response_heldout"], post_h, data["va"], data["avh"])
    sub_delta = np.asarray(after_sub) - np.asarray(before_sub)
    pmask, cmask = pre_val == parent, pre_val == child
    parent_delta = (base.nrmse(data["va"]["response_heldout"][pmask], post_h[pmask]) -
                    base.nrmse(data["va"]["response_heldout"][pmask], keep_h[pmask])) if pmask.any() else 0.0
    child_keep = base.nrmse(data["va"]["response_heldout"][cmask], keep_h[cmask]) if cmask.any() else 0.0
    child_post = base.nrmse(data["va"]["response_heldout"][cmask], post_h[cmask]) if cmask.any() else 0.0
    finite = all(np.isfinite(x).all() for _, x, _ in post.values())
    checks = {
        "bootstrap_all_domains": max(x["upper95"] for x in boot.values()) < c["global_delta_nrmse"],
        "max_subgroup_delta": float(sub_delta.max()) < c["subgroup_delta_nrmse"],
        "parent_retention": float(parent_delta) < c["parent_old_delta_nrmse"],
        "child_assimilation": float(child_post) <= float(child_keep) + c["child_old_slack_nrmse"],
        "finite": bool(finite),
    }
    return {"domains": boot, "max_subgroup_delta_nrmse": float(sub_delta.max()),
            "subgroup_delta_nrmse": sub_delta, "parent_old_delta_nrmse": float(parent_delta),
            "child_keep_nrmse": float(child_keep), "child_absorb_nrmse": float(child_post),
            "checks": checks, "pass": bool(all(checks.values()))}


def structured(bank, a, b, alive, val_a, data, c):
    m = adp9.metrics(alive, val_a, bank, a, b, data["va"], data["hv"], data["av"], data["avh"], c)
    cont = np.asarray(m["family_contingency"])
    nonempty = cont.sum(1) > 0
    purity = cont.max(1) / np.maximum(cont.sum(1), 1)
    m["major_candidate_purity"] = purity[nonempty].tolist()
    return m


def train_absorption(pre_bank, pre_a, pre_b, alive, train_a, val_a, child, parent, data, c,
                     anchor_weight, tag, out):
    bank, a, b = clone_model(pre_bank, pre_a, pre_b)
    old_bank, old_a, old_b = clone_model(pre_bank, pre_a, pre_b)
    for p in old_bank.parameters(): p.requires_grad_(False)
    old_a.requires_grad_(False); old_b.requires_grad_(False)
    # Shared trunk is frozen. Only the candidate-specific adapter tensor receives a masked gradient.
    for p in bank.parameters(): p.requires_grad_(False)
    bank.adapters.requires_grad_(True); a.requires_grad_(True); b.requires_grad_(True)
    adapter0, a0, b0 = bank.adapters.detach().clone(), a.detach().clone(), b.detach().clone()
    opt = torch.optim.AdamW([
        {"params": [bank.adapters], "lr": c["mechanism_learning_rate"], "weight_decay": c["weight_decay"]},
        {"params": [a, b], "lr": c["coordinate_learning_rate"], "weight_decay": 0.0},
    ])
    cf, pf = np.flatnonzero(train_a == child), np.flatnonzero(train_a == parent)
    pv = np.flatnonzero(val_a == parent)
    if not len(cf) or not len(pf): raise RuntimeError("child or parent has no train family")
    R = data["atr"].shape[1]
    # Anchor observations: old-parent train plus old-parent validation under observed and held-out actions.
    anchor_S = np.concatenate([np.repeat(data["tr"]["S"][pf], R, axis=0),
                               np.repeat(data["va"]["S"][pv], 2 * R, axis=0)])
    anchor_alpha = np.concatenate([data["atr"][pf].reshape(-1), data["av"][pv].reshape(-1),
                                   data["avh"][pv].reshape(-1)]).astype(np.float32)
    rng = np.random.default_rng(c["optimization_seed"] * 1000003 + child * 1009 + parent * 9173 + sum(map(ord, tag)))
    telemetry, started = [], time.time()
    half = c["family_batch_size"] // 2
    steps = max(1, math.ceil(max(len(cf), len(pf)) / half))
    for epoch in range(1, c["absorption_epochs"] + 1):
        losses = []
        for _ in range(steps):
            ci = rng.choice(cf, half, replace=len(cf) < half)
            pi = rng.choice(pf, half, replace=len(pf) < half)
            def response_loss(fi):
                St = torch.from_numpy(np.repeat(data["tr"]["S"][fi], R, axis=0)).to(DEVICE)
                at = torch.from_numpy(data["atr"][fi].reshape(-1)).to(DEVICE)
                dt = torch.from_numpy(data["tr"]["response_train"][fi].reshape(-1, 3, 2)).to(DEVICE)
                vv = torch.zeros((len(fi) * R, 5), device=DEVICE)
                vv[:, parent] = a[parent] * at + b[parent]
                return (bank.raw_effects(St, vv)[:, parent] - dt).square().mean()
            lc, lp = response_loss(ci), response_loss(pi)
            ai = rng.choice(len(anchor_alpha), c["family_batch_size"], replace=len(anchor_alpha) < c["family_batch_size"])
            St = torch.from_numpy(anchor_S[ai]).to(DEVICE)
            at = torch.from_numpy(anchor_alpha[ai]).to(DEVICE)
            vn = torch.zeros((len(ai), 5), device=DEVICE); vo = torch.zeros_like(vn)
            vn[:, parent] = a[parent] * at + b[parent]
            vo[:, parent] = old_a[parent] * at + old_b[parent]
            with torch.no_grad(): target = old_bank.raw_effects(St, vo)[:, parent]
            la = (bank.raw_effects(St, vn)[:, parent] - target).square().mean()
            loss = lc + c["lambda_replay"] * lp + anchor_weight * la
            opt.zero_grad(set_to_none=True); loss.backward()
            with torch.no_grad():
                mask = torch.zeros_like(bank.adapters.grad); mask[parent] = 1
                bank.adapters.grad.mul_(mask)
                a.grad[:parent].zero_(); a.grad[parent+1:].zero_()
                b.grad[:parent].zero_(); b.grad[parent+1:].zero_()
            torch.nn.utils.clip_grad_norm_([bank.adapters, a, b], c["gradient_clip"])
            opt.step()
            with torch.no_grad():
                keep = torch.arange(5, device=DEVICE) != parent
                bank.adapters[keep] = adapter0[keep]; a[keep] = a0[keep]; b[keep] = b0[keep]
            losses.append((float(lc.detach()), float(lp.detach()), float(la.detach()), float(loss.detach())))
        if epoch % 4 == 0:
            new_alive = [x for x in alive if x != child]
            ta, _, va, _ = assignments(bank, a, b, new_alive, data)
            child_pred = hard_predictions(bank, a, b, np.full(len(cf), parent),
                                          {"S": data["tr"]["S"][cf]}, data["atr"][cf])
            parent_pred = hard_predictions(bank, a, b, np.full(len(pf), parent),
                                           {"S": data["tr"]["S"][pf]}, data["atr"][pf])
            global_pred = hard_predictions(bank, a, b, va, data["va"], data["av"])
            telemetry.append({"epoch": epoch, "child_nrmse": base.nrmse(data["tr"]["response_train"][cf], child_pred),
                              "parent_old_nrmse": base.nrmse(data["tr"]["response_train"][pf], parent_pred),
                              "anchor_mse": float(np.mean([x[2] for x in losses])),
                              "global_validation_nrmse": base.nrmse(data["va"]["response_train"], global_pred),
                              "child_loss": float(np.mean([x[0] for x in losses])),
                              "parent_replay_loss": float(np.mean([x[1] for x in losses])),
                              "total_loss": float(np.mean([x[3] for x in losses]))})
            print(f"[{tag}] ep={epoch:02d} child={telemetry[-1]['child_nrmse']:.4f} "
                  f"parent={telemetry[-1]['parent_old_nrmse']:.4f} val={telemetry[-1]['global_validation_nrmse']:.4f}", flush=True)
    new_alive = [x for x in alive if x != child]
    ta, tc, va, vc = assignments(bank, a, b, new_alive, data)
    dump(out / "learning_curve.json", telemetry)
    return bank, a, b, new_alive, ta, tc, va, vc, telemetry, time.time() - started


def frozen_delete(pre_bank, pre_a, pre_b, alive, pre_val, child, parent, data, c, out, seed):
    new_alive = [x for x in alive if x != child]
    ta, tc, va, vc = assignments(pre_bank, pre_a, pre_b, new_alive, data)
    cmp = compare_models(pre_bank, pre_a, pre_b, alive, pre_val, pre_bank, pre_a, pre_b,
                         new_alive, va, child, parent, data, c, seed)
    result = {"operation": "frozen_delete", "child": child, "parent": parent,
              "alive_after": new_alive, "comparison": cmp, "pass": cmp["pass"]}
    dump(out / "final_metrics.json", result)
    np.savez_compressed(out / "assignments.npz", train=ta, val=va, train_cost=tc, val_cost=vc)
    return result


def trial(pre_bank, pre_a, pre_b, alive, train_a, val_a, child, parent, data, c,
          anchor_weight, tag, out, seed):
    bank, a, b, new_alive, ta, tc, va, vc, tele, seconds = train_absorption(
        pre_bank, pre_a, pre_b, alive, train_a, val_a, child, parent, data, c,
        anchor_weight, tag, out)
    cmp = compare_models(pre_bank, pre_a, pre_b, alive, val_a, bank, a, b, new_alive, va,
                         child, parent, data, c, seed)
    met = structured(bank, a, b, new_alive, va, data, c)
    result = {"operation": tag, "child": child, "parent": parent, "alive_after": new_alive,
              "assignment_change_train": float(np.mean(ta != train_a)),
              "assignment_change_val": float(np.mean(va != val_a)), "comparison": cmp,
              "structured_metrics_evaluator_only": met, "wall_seconds": seconds,
              "commit_gate_pass": cmp["pass"]}
    dump(out / "final_metrics.json", result)
    np.savez_compressed(out / "assignments.npz", train=ta, val=va, train_cost=tc, val_cost=vc)
    torch.save({"bank": bank.state_dict(), "coordinate_a": a.detach().cpu(),
                "coordinate_b": b.detach().cpu(), "alive_indices": new_alive}, out / "model.pt")
    return result, (bank, a, b, new_alive, ta, tc, va, vc)


def run_scan(seed=1, force=False):
    final = OUT / f"seed_{seed:03d}" / "pair_scan" / "pair_scan.json"
    if final.exists() and not force: return read(final)
    c, src, bank, a, b, saved, alive, data, audit = load_source(seed)
    ta, tc, va, vc = assignments(bank, a, b, alive, data)
    rows, proposals = pair_scan(alive, ta, tc, va, vc, c)
    result = {"seed": seed, "alive": alive, "learner_visible_only": True,
              "source_saved_assignment_agreement": {"train": float(np.mean(ta == saved["train"])),
                                                     "val": float(np.mean(va == saved["val"]))},
              "all_directional_pairs": rows, "passing_proposals": proposals}
    dump(final, result); dump(final.parent.parent / "baseline" / "source_audit.json", audit)
    np.savez_compressed(final.parent / "assignments_costs.npz", train=ta, val=va, train_cost=tc, val_cost=vc)
    print(json.dumps({"seed": seed, "proposal_count": len(proposals),
                      "top": [x["pair"] for x in proposals[:3]]}, indent=2), flush=True)
    return result


def run_variant(variant, seed=1, force=False):
    root = OUT / f"seed_{seed:03d}"
    final = root / f"{variant}_final.json"
    if final.exists() and not force: return read(final)
    c, src, bank, a, b, saved, alive, data, audit = load_source(seed)
    base.seed_everything(seed + 14000)
    ta, tc, va, vc = assignments(bank, a, b, alive, data)
    initial_val = va.copy(); initial_alive = alive.copy(); cycles = []
    if variant == "A4":
        rows, props = pair_scan(alive, ta, tc, va, vc, c)
        if not props: result = {"variant": variant, "seed": seed, "pass": False, "stop_reason": "no_parent_proposal"}
        else:
            p = props[0]; fr = frozen_delete(bank, a, b, alive, va, p["child"], p["parent"], data, c,
                                             root / "cycle_00" / "A4_frozen_delete", seed + 400)
            result = {"variant": variant, "seed": seed, "selected_pair": p, "frozen": fr,
                      "pass": fr["pass"], "expected_negative_pass": not fr["pass"]}
        dump(final, result); return result
    if variant == "A3":
        rows, props = pair_scan(alive, ta, tc, va, vc, c)
        if not props: result = {"variant": variant, "seed": seed, "pass": False, "stop_reason": "no_parent_proposal"}
        else:
            p = props[0]; child, proposed = p["child"], p["parent"]
            choices = [r for r in rows if r["child"] == child and r["parent"] != proposed]
            wrong = max(choices, key=lambda r: -1e30 if r["val_gap"]["median"] is None else r["val_gap"]["median"])
            res, _ = trial(bank, a, b, alive, ta, va, child, wrong["parent"], data, c,
                           c["lambda_anchor"], "A3_wrong_parent", root / "cycle_00" / "A3_wrong_parent", seed + 300)
            result = {"variant": variant, "seed": seed, "proposal_pair": p, "wrong_parent_pair": wrong,
                      "trial": res, "pass": res["commit_gate_pass"],
                      "stop_b_triggered": res["commit_gate_pass"]}
        dump(final, result); return result
    anchor = c["lambda_anchor"] if variant == "A1" else 0.0
    commit_count, stop_reason = 0, None
    for cycle in range(c["max_cycles"]):
        rows, props = pair_scan(alive, ta, tc, va, vc, c)
        croot = root / f"cycle_{cycle:02d}" / ("A1_constrained" if variant == "A1" else "A2_no_anchor")
        dump(croot / "pair_scan.json", {"all_directional_pairs": rows, "passing_proposals": props})
        if not props: stop_reason = "no_parent_proposal"; break
        committed = False; attempts = []
        for rank, p in enumerate(props[:c["max_proposals_per_cycle"]], 1):
            out = croot / f"proposal_{rank:02d}_{p['pair'].replace('->','_to_')}"
            frozen = frozen_delete(bank, a, b, alive, va, p["child"], p["parent"], data, c,
                                   out / "A0_frozen_delete", seed * 10000 + cycle * 100 + rank)
            res, state = trial(bank, a, b, alive, ta, va, p["child"], p["parent"], data, c, anchor,
                               variant, out / "trial", seed * 20000 + cycle * 100 + rank)
            attempt = {"rank": rank, "proposal": p, "frozen_delete": frozen, "trial": res,
                       "commit": bool(res["commit_gate_pass"]), "rollback": not res["commit_gate_pass"]}
            attempts.append(attempt); dump(out / "decision.json", attempt)
            if res["commit_gate_pass"]:
                bank, a, b, alive, ta, tc, va, vc = state
                # Frozen audit is a pure recomputation; it must pass the same gate exactly.
                audit_cmp = compare_models(bank, a, b, alive, va, bank, a, b, alive, va,
                                           p["parent"], p["parent"], data, c, seed + 900000 + cycle)
                attempt["post_commit_frozen_audit"] = audit_cmp
                committed = audit_cmp["pass"]
                if committed: commit_count += 1
                dump(out / "decision.json", attempt)
                break
        cycles.append({"cycle": cycle, "alive_before": len(alive) + (1 if committed else 0),
                       "attempts": attempts, "committed": committed, "alive_after": len(alive)})
        if not committed: stop_reason = "top_proposals_failed"; break
    if stop_reason is None: stop_reason = "max_cycles"
    final_metrics = structured(bank, a, b, alive, va, data, c)
    assignment_change = float(np.mean(va != initial_val))
    fg = c["final_gate"]
    final_checks = {
        "family_accuracy": final_metrics["family_hungarian_accuracy"] > fg["family_accuracy"],
        "fragmentation": final_metrics["mean_fragmentation"] < fg["mean_fragmentation"],
        "assignment_change": assignment_change < fg["assignment_change"],
        "iid": final_metrics["iid_heldout_nrmse"] < fg["iid_nrmse"],
        "structured": max(final_metrics["cross_state_nrmse"], final_metrics["cross_realization_nrmse"],
                          final_metrics["cross_base_nrmse"]) < fg["structured_nrmse"],
        "functional_evaluator_only": min(final_metrics["best_functional_r2_per_gt"]) > fg["min_best_functional_r2"],
        "purity_evaluator_only": min(final_metrics["major_candidate_purity"]) > fg["major_candidate_purity"],
    }
    result = {"variant": variant, "seed": seed, "source_retrained": False,
              "initial_alive": initial_alive, "final_alive": alive, "commit_count": commit_count,
              "cycles": cycles, "stop_reason": stop_reason, "assignment_change_from_source": assignment_change,
              "final_metrics": final_metrics, "final_gate_checks": final_checks,
              "final_gate_pass": bool(all(final_checks.values())),
              "stage_a_pass": bool(commit_count >= 1 and all(final_checks.values())),
              "strong_pass": bool(commit_count >= 1 and len(alive) == 3 and all(final_checks.values()))}
    dump(final, result); dump(root / "baseline" / "source_audit.json", audit)
    final_dir = root / "final" / variant; final_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"bank": bank.state_dict(), "coordinate_a": a.detach().cpu(), "coordinate_b": b.detach().cpu(),
                "alive_indices": alive}, final_dir / "model.pt")
    np.savez_compressed(final_dir / "assignments.npz", train=ta, val=va, train_cost=tc, val_cost=vc)
    print(json.dumps({"variant": variant, "commits": commit_count, "alive": alive,
                      "final_gate": result["final_gate_pass"], "stop": stop_reason}, indent=2), flush=True)
    return result


def smoke():
    c, src, bank, a, b, saved, alive, data, audit = load_source(1)
    ta, tc, va, vc = assignments(bank, a, b, alive, data)
    rows, props = pair_scan(alive, ta, tc, va, vc, c)
    checks = {"cuda_or_cpu_available": str(DEVICE), "checkpoint_exists": True,
              "train_shape": list(tc.shape), "val_shape": list(vc.shape),
              "finite_cost": bool(np.isfinite(tc).all() and np.isfinite(vc).all()),
              "proposal_count": len(props), "top_proposals": [x["pair"] for x in props[:3]]}
    dump(OUT / "smoke" / "final_metrics.json", checks)
    print(json.dumps(checks, indent=2), flush=True)
    return checks


def stage_a(force=False):
    scan = run_scan(1, force)
    a1 = run_variant("A1", 1, force)
    # Controls are diagnostic and still run after STOP-A; Stage B is what STOP-A forbids.
    a2 = run_variant("A2", 1, force)
    a3 = run_variant("A3", 1, force)
    a4 = run_variant("A4", 1, force)
    result = {"scan": {"proposal_count": len(scan["passing_proposals"])}, "A1": a1, "A2": a2,
              "A3": a3, "A4": a4, "stop_A": a1.get("commit_count", 0) == 0,
              "stop_B": a3.get("stop_b_triggered", False),
              "stage_b_authorized": bool(a1.get("strong_pass", False) and not a3.get("stop_b_triggered", False))}
    dump(OUT / "seed_001" / "stage_a_summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["smoke", "scan", "proof", "stage_a"], default="stage_a")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--variant", choices=["A1", "A2", "A3", "A4"], default="A1")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.stage == "smoke": smoke()
    elif args.stage == "scan": run_scan(args.seed, args.force)
    elif args.stage == "proof": run_variant(args.variant, args.seed, args.force)
    else: stage_a(args.force)


if __name__ == "__main__":
    main()
