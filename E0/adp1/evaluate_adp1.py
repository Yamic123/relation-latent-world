"""Independent hidden-GT evaluation for E0-ADP1.

This module is the only ADP1 entry point that reads ``hidden_gt.npz`` / the
ground-truth world.  It computes the prediction, participation, instance-effect,
functional-recovery, redundancy, leave-one-out and composition metrics for both
the exact solver (discovery result) and the amortized student.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from adp1._utils import DEVICE, dump_json, load_assignment, to_tensor
from adp1.exact_e_step import exact_support_e_step, freeze_bank
from adp1 import amortize_qeta
from diagnostics import common as diag

import run_e0 as base  # noqa: E402

SPLITS = ["train", "val", "test_iid", "test_context", "test_combination_101"]


class _BankOnly:
    def __init__(self, bank: torch.nn.Module) -> None:
        self.bank = bank


@torch.no_grad()
def _bank_predictions(bank: torch.nn.Module, S: np.ndarray, m: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raw_parts = []
    for start in range(0, len(S), 2048):
        end = min(start + 2048, len(S))
        raw_parts.append(bank.raw_effects(to_tensor(S[start:end]), to_tensor(v[start:end])).cpu().numpy())
    raw = np.concatenate(raw_parts, axis=0)  # [N,5,3,2]
    effects = m[:, :, None, None].astype(np.float32) * raw
    return effects, effects.sum(axis=1)


def _hungarian(matrix: np.ndarray) -> Dict[int, int]:
    rows, cols = linear_sum_assignment(-np.nan_to_num(matrix, nan=-1e6))
    return {int(gt): int(learned) for learned, gt in zip(rows, cols)}


def _structural_on_split(
    bank: torch.nn.Module,
    S: np.ndarray,
    delta: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    hidden: Mapping[str, np.ndarray],
    mapping: Dict[int, int],
    functional_matrix: Optional[np.ndarray] = None,
    instance_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    effects, prediction = _bank_predictions(bank, S, m, v)
    m_gt = hidden["m_gt"]
    learned_3 = np.column_stack([m[:, mapping[gt]] for gt in range(3)])
    exact_support = float(np.mean(np.all(learned_3 == m_gt, axis=1)))
    participation = {}
    for gt in range(3):
        participation[f"Z{gt + 1}"] = base.binary_metrics(m_gt[:, gt], m[:, mapping[gt]])
    f1_mean = float(np.mean([participation[f"Z{gt + 1}"]["f1"] for gt in range(3)]))

    inst_matched = None
    if instance_matrix is not None:
        inst_matched = [float(instance_matrix[mapping[gt], gt]) for gt in range(3)]
    func_matched = None
    if functional_matrix is not None:
        func_matched = [float(functional_matrix[mapping[gt], gt]) for gt in range(3)]

    matched_learned = set(mapping.values())
    redundant = [l for l in range(m.shape[1]) if l not in matched_learned]
    usage = m.mean(axis=0)
    redundant_usage = float(np.max(usage[redundant])) if redundant else 0.0

    return {
        "prediction_nrmse": base.nrmse(delta, prediction),
        "exact_support_accuracy": exact_support,
        "participation_f1_mean": f1_mean,
        "participation": participation,
        "instance_effect_r2_matched": inst_matched,
        "instance_effect_r2_min": min(inst_matched) if inst_matched else None,
        "functional_r2_matched": func_matched,
        "functional_r2_min": min(func_matched) if func_matched else None,
        "candidate_usage": usage.tolist(),
        "expected_active": float(m.sum(axis=1).mean()),
        "redundant_indices_1_based": [l + 1 for l in redundant],
        "redundant_usage_max": redundant_usage,
        "_effects": effects,
        "_prediction": prediction,
    }


def _leave_one_out(delta: np.ndarray, effects: np.ndarray) -> Dict[str, Any]:
    full_mse = float(np.mean((delta - effects.sum(axis=1)) ** 2))
    gains = []
    for l in range(effects.shape[1]):
        ablated = effects.sum(axis=1) - effects[:, l]
        gains.append(float(np.mean((delta - ablated) ** 2) - full_mse))
    return {"full_mse": full_mse, "gains": gains, "all_positive": all(g > 0 for g in gains)}


def evaluate_exact_seed(seed: int, cfg: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Evaluate a single discovery seed's exact solver against hidden GT."""
    discovery_dir = run_dir / "discovery"
    bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
    bank.load_state_dict(torch.load(discovery_dir / "best_bank.pt", map_location=DEVICE, weights_only=True))
    freeze_bank(bank)

    teacher_train = load_assignment(discovery_dir / "assignments" / "best_train.npz")
    teacher_val = load_assignment(discovery_dir / "assignments" / "best_val.npz")

    e_cfg = cfg["e_step"]
    pred_nrmse: Dict[str, float] = {}
    latents: Dict[str, Dict[str, np.ndarray]] = {}
    for split in SPLITS:
        S, _, delta = base.load_visible("e0a", split, "identity")
        if split == "train":
            latents[split] = teacher_train
        elif split == "val":
            latents[split] = teacher_val
        else:
            latents[split] = exact_support_e_step(bank, S, delta, e_cfg, prev_v=None, round_id=0, opt_seed=seed)
        effects, prediction = _bank_predictions(bank, S, latents[split]["m"], latents[split]["v"])
        pred_nrmse[split] = base.nrmse(delta, prediction)
        latents[split]["_prediction"] = prediction
        latents[split]["_effects"] = effects

    # Functional + instance matrices on test_iid (primary alignment source).
    hidden_iid = base.load_hidden("e0a", "test_iid")
    v_iid = latents["test_iid"]["v"]
    func, calibration, pairs = base.functional_matrix(_BankOnly(bank), v_iid, hidden_iid)
    mapping = _hungarian(func)
    inst_matrix = diag.instance_r2_matrix(latents["test_iid"]["_effects"], hidden_iid["e_gt"])

    S_iid, _, delta_iid = base.load_visible("e0a", "test_iid", "identity")
    iid_struct = _structural_on_split(
        bank, S_iid, delta_iid, latents["test_iid"]["m"], v_iid, hidden_iid, mapping,
        functional_matrix=func, instance_matrix=inst_matrix,
    )
    hidden_ctx = base.load_hidden("e0a", "test_context")
    S_ctx, _, delta_ctx = base.load_visible("e0a", "test_context", "identity")
    ctx_struct = _structural_on_split(
        bank, S_ctx, delta_ctx, latents["test_context"]["m"], latents["test_context"]["v"],
        hidden_ctx, mapping,
    )
    hidden_101 = base.load_hidden("e0a", "test_combination_101")
    S_101, _, delta_101 = base.load_visible("e0a", "test_combination_101", "identity")
    # test_101 is a constant pattern [1,0,1]; per-mechanism F1/AUROC are degenerate.
    m101 = latents["test_combination_101"]["m"]
    learned_3_101 = np.column_stack([m101[:, mapping[gt]] for gt in range(3)])
    pattern_exact = float(np.mean(np.all(learned_3_101 == hidden_101["m_gt"], axis=1)))
    active_inactive_acc = {
        f"Z{gt + 1}": float(np.mean(learned_3_101[:, gt] == hidden_101["m_gt"][:, gt]))
        for gt in range(3)
    }

    matched_learned = set(mapping.values())
    redundant = [l for l in range(cfg["m_max"]) if l not in matched_learned]
    usage = latents["test_iid"]["m"].mean(axis=0)
    redundant_usage = float(np.max(usage[redundant])) if redundant else 0.0

    loo = _leave_one_out(delta_iid, latents["test_iid"]["_effects"])

    inst_matched = iid_struct["instance_effect_r2_matched"]
    func_matched = iid_struct["functional_r2_matched"]

    stable = bool((discovery_dir / "stability.json").exists() and
                  _load_stability(discovery_dir / "stability.json")["discovery_stable"])

    passes = {
        "predictive_iid": pred_nrmse["test_iid"] < 0.05,
        "predictive_combination": pred_nrmse["test_combination_101"] < 0.05,
        "participation": iid_struct["participation_f1_mean"] > 0.9,
        "instance_effect": min(inst_matched) > 0.9,
        "functional": min(func_matched) > 0.9,
        "redundancy": redundant_usage < 0.1,
        "discovery_stable": stable,
    }
    metrics = {
        "experiment": "E0-ADP1",
        "optimization_seed": seed,
        "prediction_nrmse": pred_nrmse,
        "alignment_gt_to_learned": {str(k + 1): v + 1 for k, v in mapping.items()},
        "redundant_indices_1_based": [x + 1 for x in redundant],
        "redundant_usage_max": redundant_usage,
        "test_iid": {k: v for k, v in iid_struct.items() if not k.startswith("_")},
        "test_context": {k: v for k, v in ctx_struct.items() if not k.startswith("_")},
        "test_101": {
            "prediction_nrmse": pred_nrmse["test_combination_101"],
            "pattern_exact_accuracy": pattern_exact,
            "active_inactive_accuracy": active_inactive_acc,
            "note": "Z2 is constant-inactive in test_101; per-mechanism F1/AUROC are N/A",
        },
        "functional_r2_matrix": func.tolist(),
        "functional_r2_matched": func_matched,
        "functional_r2_min": min(func_matched),
        "instance_effect_r2_matrix": inst_matrix.tolist(),
        "instance_effect_r2_matched": inst_matched,
        "instance_effect_r2_min": min(inst_matched),
        "leave_one_candidate_out": loo,
        "calibration": calibration,
        "pass_components": passes,
        "pass": bool(all(passes.values())),
    }
    dump_json(run_dir / "discovery" / "metrics_exact.json", metrics)
    return metrics


def _load_stability(path: Path) -> Dict[str, Any]:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_amortized_seed(
    seed: int, cfg: Mapping[str, Any], run_dir: Path, tp: str
) -> Optional[Dict[str, Any]]:
    """Evaluate one amortized student against teacher and hidden GT."""
    am_dir = run_dir / "amortization" / tp
    metrics_path = am_dir / "metrics.json"
    q = base.MechanismAbstraction(cfg["m_max"], 64, 2, 4).to(DEVICE)
    q.load_state_dict(torch.load(am_dir / "qeta_best.pt", map_location=DEVICE, weights_only=True))
    q.eval()

    bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
    bank.load_state_dict(torch.load(run_dir / "discovery" / "best_bank.pt", map_location=DEVICE, weights_only=True))
    freeze_bank(bank)

    teacher_train = load_assignment(run_dir / "discovery" / "assignments" / "best_train.npz")
    teacher_m = teacher_train["m"].astype(np.float32)
    teacher_v = teacher_train["v"]

    S_train, p_train, _ = base.load_visible("e0a", "train", tp)
    student_out = amortize_qeta._student_output(q, S_train, p_train)
    fidelity = amortize_qeta.teacher_fidelity(student_out, teacher_m, teacher_v)

    logits = student_out["logits"]
    m_hat = (torch.sigmoid(torch.from_numpy(logits)).numpy() >= 0.5).astype(np.float32)
    v_hat = student_out["v"]

    hidden_iid = base.load_hidden("e0a", "test_iid")
    S_iid, p_iid, delta_iid = base.load_visible("e0a", "test_iid", tp)
    out_iid = amortize_qeta._student_output(q, S_iid, p_iid)
    m_hat_iid = (torch.sigmoid(torch.from_numpy(out_iid["logits"])).numpy() >= 0.5).astype(np.float32)
    v_hat_iid = out_iid["v"]
    func, _, _ = base.functional_matrix(_BankOnly(bank), v_hat_iid, hidden_iid)
    mapping = _hungarian(func)
    effects_iid, pred_iid = _bank_predictions(bank, S_iid, m_hat_iid, v_hat_iid)
    inst_matrix = diag.instance_r2_matrix(effects_iid, hidden_iid["e_gt"])
    learned_3 = np.column_stack([m_hat_iid[:, mapping[gt]] for gt in range(3)])
    f1s = [
        base.binary_metrics(hidden_iid["m_gt"][:, gt], m_hat_iid[:, mapping[gt]])["f1"]
        for gt in range(3)
    ]
    inst_matched = [float(inst_matrix[mapping[gt], gt]) for gt in range(3)]

    S_101, p_101, delta_101 = base.load_visible("e0a", "test_combination_101", tp)
    out_101 = amortize_qeta._student_output(q, S_101, p_101)
    m_hat_101 = (torch.sigmoid(torch.from_numpy(out_101["logits"])).numpy() >= 0.5).astype(np.float32)
    v_hat_101 = out_101["v"]
    _, pred_101 = _bank_predictions(bank, S_101, m_hat_101, v_hat_101)

    nrmse_iid = base.nrmse(delta_iid, pred_iid)
    nrmse_101 = base.nrmse(delta_101, pred_101)
    f1_mean = float(np.mean(f1s))

    passes = {
        "teacher_support_f1": fidelity["teacher_support_f1_mean"] > 0.95,
        "teacher_support_exact": fidelity["teacher_support_exact_accuracy"] > 0.90,
        "teacher_active_v_r2": fidelity["teacher_active_v_r2"] > 0.95,
        "gt_participation_f1": f1_mean > 0.90,
        "gt_instance_effect_r2_min": min(inst_matched) > 0.90,
        "iid_nrmse": nrmse_iid < 0.05,
        "combination_nrmse": nrmse_101 < 0.05,
    }
    metrics = {
        "experiment": "E0-ADP1-amortization",
        "tp": tp,
        "optimization_seed": seed,
        "teacher_fidelity": fidelity,
        "prediction_nrmse": {"test_iid": nrmse_iid, "test_combination_101": nrmse_101},
        "participation_f1_mean": f1_mean,
        "instance_effect_r2_min": min(inst_matched),
        "alignment_gt_to_learned": {str(k + 1): v + 1 for k, v in mapping.items()},
        "pass_components": passes,
        "pass": bool(all(passes.values())),
    }
    dump_json(metrics_path, metrics)
    return metrics
