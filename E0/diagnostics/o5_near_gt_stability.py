from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from common import (
    DEVICE,
    OUTPUT_ROOT,
    RunLogger,
    base,
    binary_metrics,
    dump_config,
    dump_json,
    instance_r2_matrix,
    load_all,
    metadata,
    nrmse,
    seed_record,
)


BASE_CONFIG: Dict[str, Any] = {
    "token_dim": 64,
    "attention_heads": 4,
    "cross_attention_layers": 2,
    "batch_size": 512,
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "max_epochs": 200,
    "steps_per_epoch": 64,
    "gradient_clip": 1.0,
    "lambda_participation": 1e-3,
    "temperature_start": 1.0,
    "temperature_end": 0.35,
    "metric_interval_epochs": 10,
    "metric_subset": 5000,
    "GT_supervision_after_initialization": False,
}


class NearGTModel(nn.Module):
    def __init__(self, m_max: int) -> None:
        super().__init__()
        self.m_max = m_max
        self.abstraction = base.MechanismAbstraction(m_max, 64, 2, 4)
        self.bank = base.SetMechanismBank(m_max, 64, 4)
        offset = torch.zeros(m_max)
        if m_max == 5:
            offset[3:] = -8.0
        self.gate_offsets = nn.Parameter(offset)

    def forward(self, S: torch.Tensor, p: torch.Tensor, temperature: float) -> Dict[str, torch.Tensor]:
        logits, v, _ = self.abstraction(S, p)
        logits = logits + self.gate_offsets[None]
        gates, probs = base.hard_concrete(logits, self.training, temperature)
        raw = self.bank.raw_effects(S, v)
        effects = gates[:, :, None, None] * raw
        return {
            "prediction": effects.sum(dim=1), "effects": effects, "raw": raw,
            "gates": gates, "gate_probs": probs, "v": v, "logits": logits,
        }


def copy_with_leading_slots(module: nn.Module, source: Dict[str, torch.Tensor], leading_key: str) -> None:
    target = module.state_dict()
    for key, value in source.items():
        if key not in target:
            continue
        if target[key].shape == value.shape:
            target[key] = value
        elif key == leading_key and target[key].shape[1:] == value.shape[1:]:
            target[key][: value.shape[0]] = value.to(target[key].device)
        else:
            raise ValueError(f"Unexpected near-GT state mismatch for {key}: {value.shape} -> {target[key].shape}")
    module.load_state_dict(target)


def initialize(model: NearGTModel, tp: str, seed: int) -> Dict[str, str]:
    o1_path = OUTPUT_ROOT / "O1" / f"seed_{seed}" / "checkpoint.pt"
    o2_path = OUTPUT_ROOT / "O2" / tp / f"seed_{seed}" / "checkpoint.pt"
    o1_state = torch.load(o1_path, map_location="cpu", weights_only=True)
    bank_state = {key.removeprefix("bank."): value for key, value in o1_state.items() if key.startswith("bank.")}
    o2_state = torch.load(o2_path, map_location="cpu", weights_only=True)
    copy_with_leading_slots(model.bank, bank_state, "adapters")
    copy_with_leading_slots(model.abstraction, o2_state, "queries")
    return {"O1_checkpoint": str(o1_path), "O2_checkpoint": str(o2_path)}


@torch.no_grad()
def predict(model: NearGTModel, data: Dict[str, np.ndarray], batch_size: int = 2048) -> Dict[str, np.ndarray]:
    model.eval()
    parts: Dict[str, List[np.ndarray]] = {
        key: [] for key in ["prediction", "effects", "gate_probs", "v"]
    }
    for start in range(0, len(data["S"]), batch_size):
        end = min(start + batch_size, len(data["S"]))
        output = model(
            torch.from_numpy(data["S"][start:end]).to(DEVICE),
            torch.from_numpy(data["p"][start:end]).to(DEVICE), 0.35,
        )
        for key in parts:
            parts[key].append(output[key].cpu().numpy())
    return {key: np.concatenate(value) for key, value in parts.items()}


def structure_metrics(model: NearGTModel, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    output = predict(model, data)
    matrix = instance_r2_matrix(output["effects"], data["e_gt"])
    rows, cols = linear_sum_assignment(-np.nan_to_num(matrix, nan=-1e6))
    mapping = {int(gt): int(learned) for learned, gt in zip(rows, cols)}
    matched = [float(matrix[mapping[gt], gt]) for gt in range(3)]
    participation = {
        f"Z{gt + 1}": binary_metrics(data["m_gt"][:, gt], output["gate_probs"][:, mapping[gt]])
        for gt in range(3)
    }
    return {
        "prediction_nrmse": nrmse(data["delta_S"], output["prediction"]),
        "participation_f1_mean": float(np.mean([x["f1"] for x in participation.values()])),
        "participation_f1_min": float(np.min([x["f1"] for x in participation.values()])),
        "participation_auroc_mean": float(np.mean([x["auroc"] for x in participation.values()])),
        "participation": participation,
        "instance_effect_r2_matrix": matrix.tolist(),
        "instance_mapping_gt_to_learned": mapping,
        "instance_effect_r2_matched": matched,
        "instance_effect_r2_min": float(min(matched)),
        "expected_active": float(output["gate_probs"].sum(axis=1).mean()),
        "candidate_usage": output["gate_probs"].mean(axis=0).tolist(),
        "v_mean": output["v"].mean(axis=0).tolist(),
        "v_abs_mean": np.abs(output["v"]).mean(axis=0).tolist(),
        "v_std": output["v"].std(axis=0).tolist(),
    }


def final_functional(model: NearGTModel, tp: str, inferred_v: np.ndarray) -> Dict[str, Any]:
    # base.functional_matrix only relies on .bank and therefore also supports NearGTModel.
    hidden = load_all("test_iid", tp)
    matrix, calibration, _ = base.functional_matrix(model, inferred_v, hidden)
    rows, cols = linear_sum_assignment(-np.nan_to_num(matrix, nan=-1e6))
    mapping = {int(gt): int(learned) for learned, gt in zip(rows, cols)}
    matched = [float(matrix[mapping[gt], gt]) for gt in range(3)]
    return {
        "matrix": matrix.tolist(), "mapping_gt_to_learned": mapping,
        "matched": matched, "minimum_matched": float(min(matched)), "calibration": calibration,
    }


def run(m_max: int, tp: str, seed: int, force: bool = False) -> Dict[str, Any]:
    label = "O5-A" if m_max == 3 else "O5-B"
    run_dir = OUTPUT_ROOT / "O5" / f"m{m_max}" / tp / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**BASE_CONFIG, "experiment": label, "m_max": m_max, "tp": tp,
              "extra_candidate_initial_gate_offset": -8.0 if m_max == 5 else None}
    dump_config(run_dir / "config.yaml", config)
    dump_json(run_dir / "seed.json", seed_record(seed))
    dump_json(run_dir / "run_metadata.json", metadata(label, seed, config))
    logger = RunLogger(run_dir / "stdout.log")
    base.seed_everything(seed + m_max * 10_000)
    model = NearGTModel(m_max).to(DEVICE)
    sources = initialize(model, tp, seed)
    train_data = load_all("train", tp)
    iid_data = load_all("test_iid", tp)
    metric_data = {key: value[: config["metric_subset"]] for key, value in iid_data.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed + (5_030_000 if m_max == 3 else 5_050_000))
    curves: List[Dict[str, float]] = []
    structural: List[Dict[str, Any]] = []
    usage: List[Dict[str, Any]] = []
    started = time.time()
    initial = structure_metrics(model, metric_data)
    structural.append({"epoch": 0, "stage": "before_unsupervised_training", **initial})
    usage.append({"epoch": 0, "stage": "before_unsupervised_training",
                  "candidate_usage": initial["candidate_usage"], "expected_active": initial["expected_active"]})
    logger.log(
        f"{label} m={m_max} tp={tp} seed={seed} device={DEVICE} initial_nrmse={initial['prediction_nrmse']:.5f} "
        f"initial_f1={initial['participation_f1_mean']:.4f} initial_instance_r2={initial['instance_effect_r2_min']:.4f}"
    )
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        fraction = (epoch - 1) / max(config["max_epochs"] - 1, 1)
        temperature = config["temperature_start"] * (
            config["temperature_end"] / config["temperature_start"]
        ) ** fraction
        loss_sum = effect_sum = active_sum = 0.0
        for _ in range(config["steps_per_epoch"]):
            idx = rng.integers(0, len(train_data["S"]), config["batch_size"])
            output = model(
                torch.from_numpy(train_data["S"][idx]).to(DEVICE),
                torch.from_numpy(train_data["p"][idx]).to(DEVICE), temperature,
            )
            target = torch.from_numpy(train_data["delta_S"][idx]).to(DEVICE)
            effect_loss = F.mse_loss(output["prediction"], target)
            expected_active = output["gate_probs"].sum(dim=1).mean()
            loss = effect_loss + config["lambda_participation"] * expected_active
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach()); effect_sum += float(effect_loss.detach()); active_sum += float(expected_active.detach())
        record = {
            "epoch": epoch, "loss": loss_sum / config["steps_per_epoch"],
            "effect_mse": effect_sum / config["steps_per_epoch"],
            "expected_active": active_sum / config["steps_per_epoch"], "temperature": temperature,
        }
        curves.append(record)
        if epoch % config["metric_interval_epochs"] == 0 or epoch == config["max_epochs"]:
            current = structure_metrics(model, metric_data)
            structural.append({"epoch": epoch, "stage": "unsupervised", **current})
            usage.append({"epoch": epoch, "stage": "unsupervised",
                          "candidate_usage": current["candidate_usage"], "expected_active": current["expected_active"]})
            logger.log(
                f"epoch={epoch:03d} loss={record['loss']:.6f} nrmse={current['prediction_nrmse']:.5f} "
                f"f1={current['participation_f1_mean']:.4f} instance_r2={current['instance_effect_r2_min']:.4f} "
                f"active={current['expected_active']:.3f} usage={np.round(current['candidate_usage'], 3).tolist()}"
            )
            dump_json(run_dir / "structural_metrics_over_time.json", structural)
            dump_json(run_dir / "candidate_usage_over_time.json", usage)
        dump_json(run_dir / "training_curves.json", curves)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    final_splits = {
        split: structure_metrics(model, load_all(split, tp))
        for split in ["test_iid", "test_context", "test_combination_101"]
    }
    final_iid_output = predict(model, iid_data)
    functional = final_functional(model, tp, final_iid_output["v"])
    final = final_splits["test_iid"]
    metrics = {
        "experiment": label, "m_max": m_max, "tp": tp, "optimization_seed": seed,
        "initialization_sources": sources, "initial_metrics_subset": initial,
        "final_splits": final_splits, "final_functional_r2": functional,
        "extra_candidates_reopened": bool(
            m_max == 5 and max(final["candidate_usage"][3:]) > 0.1
        ),
        "gt_like_structure_stable": bool(
            final["participation_f1_mean"] > 0.9 and final["instance_effect_r2_min"] > 0.9
        ),
        "wall_seconds": time.time() - started,
    }
    dump_json(metrics_path, metrics)
    logger.log(json.dumps(metrics, ensure_ascii=False))
    logger.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-max", choices=["3", "5", "both"], default="both")
    parser.add_argument("--tp", choices=["identity", "orthogonal"], default="identity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    values = [3, 5] if args.m_max == "both" else [int(args.m_max)]
    results = [run(value, args.tp, args.seed, args.force) for value in values]
    dump_json(OUTPUT_ROOT / "O5" / "summary.json", {"experiment": "O5", "runs": results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
