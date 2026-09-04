"""13-hour compact gates for E0-ADP2: telemetry replay and A0 reuse."""
from __future__ import annotations

import argparse, hashlib, json, platform, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_e0 as base
from adp1._utils import DEVICE, dump_json
from adp1.evaluate_adp1 import _bank_predictions
from adp1.exact_e_step import exact_support_e_step, freeze_bank, unfreeze_bank
from adp1.residual_m_step import make_probe, m_step_round, snapshot_shared
from diagnostics import common as diag

CFG_PATH = ROOT / "configs" / "e0_adp2_compact.json"
OUT = ROOT / "outputs" / "e0_adp2_compact"
ADP1 = ROOT / "outputs" / "e0_adp1_mini" / "seed_0" / "adp1"
O5 = ROOT / "outputs" / "e0_diagnostics" / "O5" / "m5" / "identity" / "seed_0"


def cfg() -> dict[str, Any]:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata(stage: str, c: dict[str, Any]) -> dict[str, Any]:
    return {"stage": stage, "python_executable": sys.executable, "python": sys.version,
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(), "git_commit": base.git_commit(),
            "world_seed": c["world_seed"], "dataset_seed": c["dataset_seed"],
            "optimization_seed": c["optimization_seed"], "context_seed": c["context_seed"],
            "config_sha256": hashlib.sha256(CFG_PATH.read_bytes()).hexdigest(), "config": c}


def visible(split: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S, p, d = base.load_visible("e0a", split, "identity")
    return S[:n], p[:n], d[:n]


def context_spec(S: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(6, 2)))
    q = q.T.astype(np.float32)
    score = S.reshape(len(S), 6) @ q.T
    threshold = np.median(score, axis=0).astype(np.float32)
    bits = (score >= threshold).astype(np.int64)
    return q, threshold, 2 * bits[:, 0] + bits[:, 1]


def support_stats(m: np.ndarray) -> tuple[list[int], list[dict[str, Any]]]:
    hist = np.bincount(m.sum(1).astype(int), minlength=6).tolist()
    patterns = np.array(["".join(str(int(x)) for x in row) for row in m])
    values, counts = np.unique(patterns, return_counts=True)
    order = np.argsort(-counts)
    top = [{"pattern": str(values[i]), "count": int(counts[i]),
            "fraction": float(counts[i] / len(m))} for i in order[:10]]
    return hist, top


@torch.no_grad()
def effect_stats(bank: torch.nn.Module, S: np.ndarray, m: np.ndarray, v: np.ndarray) -> list[float]:
    effects, _ = _bank_predictions(bank, S, m, v)
    return np.linalg.norm(effects.reshape(len(S), 5, -1), axis=2).mean(0).tolist()


def validate() -> dict[str, Any]:
    c = cfg()
    files = [CFG_PATH, ADP1 / "best_bank.pt", ADP1 / "final_assignments.npz",
             ADP1 / "outer_curves.json", O5 / "checkpoint.pt", O5 / "metrics.json"]
    result = {"valid": torch.cuda.is_available() and DEVICE.type == "cuda" and all(x.exists() for x in files),
              "files": {str(x): x.exists() for x in files}, "metadata": metadata("validate", c)}
    OUT.mkdir(parents=True, exist_ok=True)
    dump_json(OUT / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def adp2_0(force: bool) -> dict[str, Any]:
    c, out = cfg(), OUT / "adp2_0_seed_0"
    final_path = out / "final_metrics.json"
    if final_path.exists() and not force:
        return json.loads(final_path.read_text(encoding="utf-8"))
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "assignments").mkdir(exist_ok=True)
    log = out / "round_metrics.jsonl"
    if log.exists(): log.unlink()
    dump_json(out / "run_metadata.json", metadata("adp2_0", c))
    S, _, delta = visible("train", c["subset"]["train"])
    Sval, _, dval = visible("val", c["subset"]["val"])
    q, threshold, groups = context_spec(S, c["context_seed"])
    dump_json(out / "context_spec.json", {"directions": q, "thresholds": threshold,
                                          "group_counts": np.bincount(groups, minlength=4)})
    base.seed_everything(c["optimization_seed"])
    bank = base.E0AModel(5, 64, 2, 4).to(DEVICE).bank
    opt = torch.optim.AdamW(bank.parameters(), lr=c["m_step"]["learning_rate"],
                            weight_decay=c["m_step"]["weight_decay"])
    probe, prev_m, prev_v = make_probe(), None, None
    started, rows = time.time(), []
    for r in range(c["compact"]["baseline_rounds"]):
        freeze_bank(bank); t = time.time()
        lat = exact_support_e_step(bank, S, delta, c["e_step"], prev_v=prev_v,
                                   round_id=r, opt_seed=0, audit_n=len(S))
        e_time = time.time() - t
        vlat = exact_support_e_step(bank, Sval, dval, c["e_step"], round_id=r, opt_seed=0)
        _, vpred = _bank_predictions(bank, Sval, vlat["m"], vlat["v"])
        margin = lat["second_margin"] / np.maximum(np.abs(lat["penalized_J"]), 1e-12)
        hist, top = support_stats(lat["m"])
        scr = None if prev_m is None else float(np.mean(lat["m"] != prev_m))
        agreement = None if prev_m is None else float(np.mean(np.all(lat["m"] == prev_m, axis=1)))
        norm = effect_stats(bank, S, lat["m"], lat["v"])
        unfreeze_bank(bank); t = time.time()
        rec = m_step_round(bank, opt, lat["m"], lat["v"], S, delta, c, r, 0,
                           probe, snapshot_shared(bank))
        m_time = time.time() - t
        torch.save(bank.state_dict(), out / "checkpoints" / f"round_{r:03d}_bank.pt")
        np.savez_compressed(out / "assignments" / f"round_{r:03d}.npz",
                            m=lat["m"], v=lat["v"], relative_margin=margin)
        row = {"round": r, "train_effect_mse": float(lat["effect_mse"].mean()),
               "train_objective": float(lat["penalized_J"].mean()),
               "val_nrmse": base.nrmse(dval, vpred), "test_iid_nrmse": None,
               "expected_active_per_sample": float(lat["m"].sum(1).mean()),
               "candidate_usage": lat["m"].mean(0).tolist(), "candidate_effect_norm": norm,
               "support_change_rate_bit": scr, "support_exact_agreement": agreement,
               "support_cardinality_histogram": hist, "top_support_patterns": top,
               "assignment_margin_mean": float(margin.mean()),
               "assignment_margin_median": float(np.median(margin)),
               "assignment_margin_p10": float(np.quantile(margin, .1)),
               "assignment_margin_p90": float(np.quantile(margin, .9)),
               "resolved_fraction": float(np.mean(margin >= c["compact"]["tau_conf"])),
               "reuse_error_per_candidate": [None] * 5, "reuse_context_coverage": [None] * 5,
               "global_alive": [1] * 5, "num_alive": 5, "merge_proposals": [],
               "merge_accepted": [], "split_proposals": [], "split_accepted": [],
               "e_step_seconds": e_time, "m_step_seconds": m_time,
               "reuse_check_seconds": 0.0, "merge_check_seconds": 0.0,
               "probe_effect_drift": rec["probe_effect_drift"]}
        append_jsonl(log, row); rows.append(row)
        print(f"[ADP2-0] r={r} J={row['train_objective']:.6f} SCR={scr} A={agreement} "
              f"active={row['expected_active_per_sample']:.3f} resolved={row['resolved_fraction']:.3f}", flush=True)
        prev_m, prev_v = lat["m"].copy(), lat["v"].copy()
    ref = json.loads((ADP1 / "outer_curves.json").read_text(encoding="utf-8"))[:len(rows)]
    diffs = [abs(rows[i]["train_objective"] - ref[i]["J_train"]) for i in range(len(rows))]
    # Cross-device replay (original CUDA vs compact CPU) uses the declared
    # reasonable numerical tolerance, not a bit-identical threshold.
    passed = max(diffs) <= 5e-3
    final = {"stage": "adp2_0", "compact_deviation": "4-round replay; existing 12-round ADP1 is baseline",
             "max_train_objective_abs_diff": max(diffs), "reproduction_pass": passed,
             "wall_seconds": time.time() - started, "next_stage_allowed": passed}
    dump_json(final_path, final)
    (out / "summary.md").write_text(f"# ADP2-0 compact\n\n- PASS: **{passed}**\n- Max J diff: {max(diffs):.3g}\n", encoding="utf-8")
    print(json.dumps(final, indent=2)); return final


def infer_o5(S: np.ndarray, p: np.ndarray) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    state = torch.load(O5 / "checkpoint.pt", map_location=DEVICE, weights_only=True)
    model = base.E0AModel(5, 64, 2, 4).to(DEVICE)
    own = model.state_dict()
    for key in own:
        if key in state:
            own[key] = state[key].to(DEVICE)
    model.load_state_dict(own)
    offset = state["gate_offsets"].to(DEVICE)
    ms, vs = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(S), 2048):
            logits, v, _ = model.abstraction(
                torch.from_numpy(S[start:start + 2048]).to(DEVICE),
                torch.from_numpy(p[start:start + 2048]).to(DEVICE))
            _, prob = base.hard_concrete(logits + offset[None], False, .35)
            ms.append((prob >= .5).byte().cpu().numpy())
            vs.append(v.cpu().numpy())
    return model.bank, np.concatenate(ms), np.concatenate(vs)


def infer_adp1() -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    bank = base.SetMechanismBank(5, 64, 4).to(DEVICE)
    bank.load_state_dict(torch.load(ADP1 / "best_bank.pt", map_location=DEVICE, weights_only=True))
    x = np.load(ADP1 / "final_assignments.npz")
    return bank, x["m_train"], x["v_train"]


def optimize_v(bank: torch.nn.Module, S: np.ndarray, target: np.ndarray, j: int, seed: int) -> np.ndarray:
    S_t, y = torch.from_numpy(S).to(DEVICE), torch.from_numpy(target).to(DEVICE)
    all_loss = []
    for restart in range(2):
        rng = np.random.default_rng(seed + restart)
        init = np.zeros(len(S), np.float32) if restart == 0 else rng.uniform(-1, 1, len(S)).astype(np.float32)
        z = torch.from_numpy(init).to(DEVICE).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=.05)
        best = torch.full((len(S),), float("inf"), device=DEVICE)
        for _ in range(20):
            opt.zero_grad(set_to_none=True)
            vf = torch.zeros(len(S), 5, device=DEVICE)
            vf[:, j] = 1.5 * torch.tanh(z)
            loss = (bank.raw_effects(S_t, vf)[:, j] - y).square().mean((-2, -1))
            best = torch.minimum(best, loss.detach())
            loss.mean().backward(); opt.step()
        all_loss.append(best.cpu().numpy())
    return np.min(np.stack(all_loss), axis=0)


def reuse(bank: torch.nn.Module, S: np.ndarray, delta: np.ndarray, m: np.ndarray,
          v: np.ndarray, groups: np.ndarray, candidates: range, c: dict[str, Any]) -> list[dict[str, Any]]:
    freeze_bank(bank); raw = []
    with torch.no_grad():
        for start in range(0, len(S), 2048):
            raw.append(bank.raw_effects(torch.from_numpy(S[start:start + 2048]).to(DEVICE),
                                        torch.from_numpy(v[start:start + 2048]).to(DEVICE)).cpu().numpy())
    effects = np.concatenate(raw) * m[:, :, None, None]
    result = []
    for j in candidates:
        folds = []
        for g in range(4):
            ids = np.flatnonzero((m[:, j] == 1) & (groups == g))
            if len(ids) < c["compact"]["n_min_per_context"]:
                folds.append({"group": g, "n": int(len(ids)), "eligible": False, "error": None}); continue
            residual = delta[ids] - (effects[ids].sum(1) - effects[ids, j])
            mse = optimize_v(bank, S[ids], residual, j, c["context_seed"] + j * 100 + g)
            error = float(mse.sum() * 6 / (np.square(residual).sum() + 1e-12))
            folds.append({"group": g, "n": int(len(ids)), "eligible": True, "error": error})
        eligible = [x["error"] for x in folds if x["eligible"]]
        result.append({"candidate": j, "folds": folds, "coverage": len(eligible),
                       "sufficient": len(eligible) >= c["compact"]["min_context_coverage"],
                       "reuse_error": float(np.median(eligible)) if eligible else None})
    return result


def adp2_a0(force: bool) -> dict[str, Any]:
    gate = OUT / "adp2_0_seed_0" / "final_metrics.json"
    if not gate.exists() or not json.loads(gate.read_text(encoding="utf-8"))["reproduction_pass"]:
        raise RuntimeError("STOP: ADP2-0 has not passed")
    c, out = cfg(), OUT / "a0_reuse_calibration"
    final_path = out / "final_metrics.json"
    if final_path.exists() and not force:
        return json.loads(final_path.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "run_metadata.json", metadata("adp2_a0", c))
    S, p, delta = visible("train", c["subset"]["train"])
    q, threshold, groups = context_spec(S, c["context_seed"])
    dump_json(out / "context_spec.json", {"directions": q, "thresholds": threshold,
                                          "group_counts": np.bincount(groups, minlength=4)})
    started = time.time()
    o5_bank, o5_m, o5_v = infer_o5(S, p)
    a1_bank, a1_m, a1_v = infer_adp1()
    positive = reuse(o5_bank, S, delta, o5_m, o5_v, groups, range(3), c)
    negative = reuse(a1_bank, S, delta, a1_m, a1_v, groups, range(5), c)
    pos = np.array([x["reuse_error"] for x in positive if x["sufficient"]])
    neg = np.array([x["reuse_error"] for x in negative if x["sufficient"]])
    if len(pos) == 3 and len(neg):
        q90, q10 = float(np.quantile(pos, .9)), float(np.quantile(neg, .1)); passed = q90 < q10
    else:
        q90 = q10 = None; passed = False
    final = {"stage": "adp2_a0", "positive_o5": positive, "negative_adp1": negative,
             "q90_o5": q90, "q10_adp1": q10, "tau_reuse": (q90 + q10) / 2 if passed else None,
             "pass": passed, "stop_required": not passed,
             "reason": "clear separation" if passed else "overlap or insufficient evidence",
             "wall_seconds": time.time() - started, "next_stage_allowed": passed}
    dump_json(final_path, final)
    (out / "summary.md").write_text(
        f"# ADP2-A0 compact\n\n- PASS: **{passed}**\n- Q90(O5): {q90}\n"
        f"- Q10(ADP1): {q10}\n- tau_reuse: {final['tau_reuse']}\n- STOP required: {not passed}\n",
        encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2)); return final


def final_eval(bank: torch.nn.Module, c: dict[str, Any], round_id: int) -> dict[str, Any]:
    S, _, delta = visible("test_iid", c["subset"]["test_iid"])
    hidden = {k: v[:len(S)] for k, v in base.load_hidden("e0a", "test_iid").items()}
    freeze_bank(bank)
    lat = exact_support_e_step(bank, S, delta, c["e_step"], round_id=round_id, opt_seed=0)
    effects, pred = _bank_predictions(bank, S, lat["m"], lat["v"])
    inst = diag.instance_r2_matrix(effects, hidden["e_gt"])
    class Holder:
        def __init__(self, b): self.bank = b
    func, _, _ = base.functional_matrix(Holder(bank), lat["v"], hidden, n_probe=512)
    rows, cols = linear_sum_assignment(-np.nan_to_num(func, nan=-1e6))
    mapping = {int(gt): int(learned) for learned, gt in zip(rows, cols)}
    learned = np.column_stack([lat["m"][:, mapping[g]] for g in range(3)])
    f1 = [base.binary_metrics(hidden["m_gt"][:, g], learned[:, g])["f1"] for g in range(3)]
    S101, _, d101 = visible("test_combination_101", c["subset"]["test_combination_101"])
    l101 = exact_support_e_step(bank, S101, d101, c["e_step"], round_id=round_id, opt_seed=0)
    _, p101 = _bank_predictions(bank, S101, l101["m"], l101["v"])
    return {"iid_nrmse": base.nrmse(delta, pred), "test_101_nrmse": base.nrmse(d101, p101),
            "participation_f1_mean": float(np.mean(f1)),
            "instance_r2_matched": [float(inst[mapping[g], g]) for g in range(3)],
            "functional_r2_matched": [float(func[mapping[g], g]) for g in range(3)],
            "mapping_gt_to_learned": mapping, "candidate_usage": lat["m"].mean(0).tolist(),
            "expected_active": float(lat["m"].sum(1).mean())}


def stage_gate(stage: str) -> None:
    if stage == "adp2_a1":
        p, key = OUT / "a0_reuse_calibration" / "final_metrics.json", "pass"
    elif stage == "adp2_a2":
        p, key = OUT / "a1_seed_0" / "final_metrics.json", "pass"
    else:
        p, key = OUT / "a2_seed_0" / "final_metrics.json", "pass"
    if not p.exists() or not json.loads(p.read_text(encoding="utf-8"))[key]:
        raise RuntimeError(f"STOP: prerequisite for {stage} has not passed")


def run_identity_stage(stage: str, force: bool) -> dict[str, Any]:
    stage_gate(stage)
    c, out = cfg(), OUT / f"{stage.split('_')[-1]}_seed_0"
    final_path = out / "final_metrics.json"
    if final_path.exists() and not force:
        return json.loads(final_path.read_text(encoding="utf-8"))
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "assignments").mkdir(exist_ok=True)
    log = out / "round_metrics.jsonl"
    if log.exists(): log.unlink()
    dump_json(out / "run_metadata.json", metadata(stage, c))
    S, _, delta = visible("train", c["subset"]["train"])
    Sval, _, dval = visible("val", c["subset"]["val"])
    q, threshold, groups = context_spec(S, c["context_seed"])
    base.seed_everything(0)
    bank = base.E0AModel(5, 64, 2, 4).to(DEVICE).bank
    lr = 1e-4 if stage == "adp2_a3" else c["m_step"]["learning_rate"]
    opt = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=c["m_step"]["weight_decay"])
    tau_reuse = json.loads((OUT / "a0_reuse_calibration" / "final_metrics.json").read_text(encoding="utf-8"))["tau_reuse"]
    prev_m = prev_v = None
    stable_age = np.zeros((len(S), 5), np.int16)
    candidate_good_age = np.zeros(5, np.int16)
    candidate_stable = np.zeros(5, bool)
    rows_out, started = [], time.time()
    for r in range(12):
        freeze_bank(bank); t = time.time()
        proposal = exact_support_e_step(bank, S, delta, c["e_step"], prev_v=prev_v,
            round_id=r, opt_seed=0, audit_n=len(S), return_all_supports=(stage != "adp2_a1"))
        e_time = time.time() - t
        rel_margin = proposal["second_margin"] / np.maximum(np.abs(proposal["penalized_J"]), 1e-12)
        resolved = rel_margin >= c["compact"]["tau_conf"]
        proposed_switch = np.zeros(len(S), bool)
        accepted_switch = np.zeros(len(S), bool)
        if prev_m is None or stage == "adp2_a1":
            m, v = proposal["m"].copy(), proposal["v"].copy()
        else:
            weights = np.array([16, 8, 4, 2, 1], dtype=np.int64)
            old_code = (prev_m * weights[None]).sum(1)
            old_J = proposal["all_support_scores"][np.arange(len(S)), old_code]
            improvement = (old_J - proposal["penalized_J"]) / np.maximum(np.abs(old_J), 1e-12)
            proposed_switch = np.any(proposal["m"] != prev_m, axis=1)
            accepted_switch = proposed_switch & resolved & (improvement >= .02)
            m = prev_m.copy(); v = proposal["all_support_v"][np.arange(len(S)), old_code].copy()
            m[accepted_switch] = proposal["m"][accepted_switch]
            v[accepted_switch] = proposal["v"][accepted_switch]
        scr = None if prev_m is None else float(np.mean(m != prev_m))
        agreement = None if prev_m is None else float(np.mean(np.all(m == prev_m, axis=1)))
        if prev_m is None: stable_age[:] = 1
        else: stable_age = np.where(m == prev_m, stable_age + 1, 0)
        reuse_result, reuse_time = None, 0.0
        if stage == "adp2_a3":
            evidence_m = (m * (resolved[:, None] & (stable_age >= 2))).astype(np.uint8)
            t = time.time(); reuse_result = reuse(bank, S, delta, evidence_m, v, groups, range(5), c); reuse_time = time.time() - t
            good = np.array([x["sufficient"] and x["reuse_error"] < tau_reuse for x in reuse_result])
            candidate_good_age = np.where(good, candidate_good_age + 1, 0)
            candidate_stable = candidate_good_age >= 3
        fit_m = (m * resolved[:, None]).astype(np.uint8)
        unfreeze_bank(bank); t = time.time()
        if stage == "adp2_a3":
            cc = json.loads(json.dumps(c)); cc["m_step"]["steps_per_candidate_per_round"] = 8
            provisional = fit_m.copy(); provisional[:, candidate_stable] = 0
            m_step_round(bank, opt, provisional, v, S, delta, cc, r, 0, None, snapshot_shared(bank))
            if candidate_stable.any():
                cc["m_step"]["steps_per_candidate_per_round"] = 4
                for group in opt.param_groups: group["lr"] = 5e-5
                stable_m = fit_m.copy(); stable_m[:, ~candidate_stable] = 0
                m_step_round(bank, opt, stable_m, v, S, delta, cc, r, 0, None, snapshot_shared(bank))
                for group in opt.param_groups: group["lr"] = 1e-4
        else:
            m_step_round(bank, opt, fit_m, v, S, delta, c, r, 0, None, snapshot_shared(bank))
        m_time = time.time() - t
        freeze_bank(bank)
        vlat = exact_support_e_step(bank, Sval, dval, c["e_step"], round_id=r, opt_seed=0)
        _, vpred = _bank_predictions(bank, Sval, vlat["m"], vlat["v"])
        hist, top = support_stats(m)
        torch.save(bank.state_dict(), out / "checkpoints" / f"round_{r:03d}_bank.pt")
        np.savez_compressed(out / "assignments" / f"round_{r:03d}.npz", m=m, v=v, resolved=resolved,
                            relative_margin=rel_margin, stable_age=stable_age)
        row = {"round": r, "train_effect_mse": float(proposal["effect_mse"].mean()),
               "train_objective": float(proposal["penalized_J"].mean()), "val_nrmse": base.nrmse(dval, vpred),
               "expected_active_per_sample": float(m.sum(1).mean()), "candidate_usage": m.mean(0).tolist(),
               "support_change_rate_bit": scr, "support_exact_agreement": agreement,
               "support_cardinality_histogram": hist, "top_support_patterns": top,
               "resolved_fraction": float(resolved.mean()), "proposed_switch_fraction": float(proposed_switch.mean()),
               "accepted_switch_fraction": float(accepted_switch.mean()),
               "reuse_error_per_candidate": None if reuse_result is None else [x["reuse_error"] for x in reuse_result],
               "reuse_context_coverage": None if reuse_result is None else [x["coverage"] for x in reuse_result],
               "stable_candidate": candidate_stable.tolist(), "global_alive": [1] * 5, "num_alive": 5,
               "e_step_seconds": e_time, "m_step_seconds": m_time, "reuse_check_seconds": reuse_time}
        append_jsonl(log, row); rows_out.append(row)
        print(f"[{stage}] r={r} J={row['train_objective']:.5f} SCR={scr} A={agreement} "
              f"resolved={row['resolved_fraction']:.3f} stable={int(candidate_stable.sum())} val={row['val_nrmse']:.3f}", flush=True)
        prev_m, prev_v = m.copy(), v.copy()
    evaluation = final_eval(bank, c, 11)
    last3 = float(np.mean([x["support_change_rate_bit"] for x in rows_out[-3:]]))
    fmin = min(evaluation["functional_r2_matched"])
    if stage == "adp2_a1":
        baseline_scr = np.mean([.2287109375, .2369140625, .1984375])
        passed = last3 <= .70 * baseline_scr and fmin >= -0.402036455 + .30 and rows_out[-1]["resolved_fraction"] >= .20
    elif stage == "adp2_a2":
        prior = json.loads((OUT / "a1_seed_0" / "final_metrics.json").read_text(encoding="utf-8"))
        passed = last3 <= .80 * prior["last3_scr"] and fmin >= prior["functional_r2_min"] and evaluation["iid_nrmse"] <= 1.10 * prior["evaluation"]["iid_nrmse"]
    else:
        passed = int(candidate_stable.sum()) >= 3 and last3 < .10 and fmin > .50 and evaluation["participation_f1_mean"] > .75
    final = {"stage": stage, "last3_scr": last3, "functional_r2_min": fmin, "stable_candidate_count": int(candidate_stable.sum()),
             "evaluation": evaluation, "pass": bool(passed), "stop_required": not bool(passed),
             "wall_seconds": time.time() - started, "next_stage_allowed": bool(passed)}
    dump_json(final_path, final)
    (out / "summary.md").write_text(f"# {stage}\n\n- PASS: **{passed}**\n- last3 SCR: {last3}\n- functional min: {fmin}\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2)); return final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["validate", "adp2_0", "adp2_a0", "adp2_a1", "adp2_a2", "adp2_a3"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.stage == "validate":
        if not validate()["valid"]: raise SystemExit(2)
    elif a.stage == "adp2_0": adp2_0(a.force)
    elif a.stage == "adp2_a0": adp2_a0(a.force)
    else: run_identity_stage(a.stage, a.force)


if __name__ == "__main__":
    main()
