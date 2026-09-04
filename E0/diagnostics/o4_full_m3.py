from __future__ import annotations

import argparse
import json
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
    aggregate_seed_metrics,
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
    "m_max": 3,
    "token_dim": 64,
    "attention_heads": 4,
    "cross_attention_layers": 2,
    "batch_size": 512,
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "max_epochs": 200,
    "steps_per_epoch": 64,
    "patience": 20,
    "gradient_clip": 1.0,
    "lambda_participation": 1e-3,
    "temperature_start": 1.0,
    "temperature_end": 0.35,
    "validation_subset": 5000,
    "only_change_from_formal_E0A": "m_max: 5 -> 3",
}


def assignment(matrix: np.ndarray) -> Dict[int, int]:
    rows, cols = linear_sum_assignment(-np.nan_to_num(matrix, nan=-1e6))
    return {int(gt): int(learned) for learned, gt in zip(rows, cols)}


def matched_values(matrix: np.ndarray, mapping: Dict[int, int]) -> List[float]:
    return [float(matrix[mapping[gt], gt]) for gt in range(3)]


def evaluate(model: base.E0AModel, tp: str) -> Dict[str, Any]:
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    prediction_nrmse: Dict[str, float] = {}
    split_instance: Dict[str, Any] = {}
    for split in base.SPLIT_SIZES:
        data = load_all(split, tp)
        out = base.predict_a(model, data["S"], data["p"])
        outputs[split] = out
        prediction_nrmse[split] = nrmse(data["delta_S"], out["prediction"])
        matrix = instance_r2_matrix(out["effects"], data["e_gt"])
        mapping = assignment(matrix)
        values = matched_values(matrix, mapping)
        split_instance[split] = {
            "matrix": matrix.tolist(), "hungarian_mapping_gt_to_learned": mapping,
            "matched": values, "minimum_matched": float(min(values)),
        }
    iid = load_all("test_iid", tp)
    functional, calibration, _ = base.functional_matrix(model, outputs["test_iid"]["v"], iid)
    functional_mapping = assignment(functional)
    functional_values = matched_values(functional, functional_mapping)
    participation: Dict[str, Any] = {}
    for gt in range(3):
        learned = functional_mapping[gt]
        participation[f"Z{gt + 1}"] = binary_metrics(
            iid["m_gt"][:, gt], outputs["test_iid"]["gate_probs"][:, learned]
        )
    f1s = [entry["f1"] for entry in participation.values()]
    aucs = [entry["auroc"] for entry in participation.values()]
    usage = outputs["test_iid"]["gate_probs"].mean(axis=0)
    passes = {
        "iid_nrmse_lt_0.05": prediction_nrmse["test_iid"] < 0.05,
        "combination_101_nrmse_lt_0.05": prediction_nrmse["test_combination_101"] < 0.05,
        "participation_f1_mean_gt_0.9": float(np.mean(f1s)) > 0.9,
        "instance_effect_r2_min_gt_0.9": split_instance["test_iid"]["minimum_matched"] > 0.9,
        "functional_r2_min_gt_0.9": min(functional_values) > 0.9,
    }
    return {
        "prediction_nrmse": prediction_nrmse,
        "participation": participation,
        "participation_f1_mean": float(np.mean(f1s)),
        "participation_f1_min": float(np.min(f1s)),
        "participation_auroc_mean": float(np.mean(aucs)),
        "instance_effect_r2": split_instance,
        "functional_r2": {
            "matrix": functional.tolist(), "hungarian_mapping_gt_to_learned": functional_mapping,
            "matched": functional_values, "minimum_matched": float(min(functional_values)),
            "calibration": calibration,
        },
        "expected_active": float(outputs["test_iid"]["gate_probs"].sum(axis=1).mean()),
        "candidate_usage": usage.tolist(),
        "pass_components": passes,
        "pass": bool(all(passes.values())),
    }


def train(tp: str, seed: int, force: bool = False) -> Dict[str, Any]:
    config = {**BASE_CONFIG, "tp": tp}
    run_dir = OUTPUT_ROOT / "O4" / tp / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(run_dir / "config.yaml", config)
    dump_json(run_dir / "seed.json", seed_record(seed))
    dump_json(run_dir / "run_metadata.json", metadata("O4", seed, config))
    logger = RunLogger(run_dir / "stdout.log")
    base.seed_everything(seed)
    train_data = load_all("train", tp)
    val_data = load_all("val", tp)
    model = base.E0AModel(3, config["token_dim"], config["cross_attention_layers"], config["attention_heads"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed + (4_010_000 if tp == "identity" else 4_020_000))
    curves: List[Dict[str, float]] = []
    best_val, stale, best_epoch = float("inf"), 0, -1
    started = time.time()
    logger.log(f"O4 full M=3 tp={tp} seed={seed} device={DEVICE}")
    for epoch in range(config["max_epochs"]):
        model.train()
        fraction = epoch / max(config["max_epochs"] - 1, 1)
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
        model.eval()
        with torch.no_grad():
            n_val = config["validation_subset"]
            val_out = base.predict_a(model, val_data["S"][:n_val], val_data["p"][:n_val])
            val_score = nrmse(val_data["delta_S"][:n_val], val_out["prediction"])
        record = {
            "epoch": epoch, "loss": loss_sum / config["steps_per_epoch"],
            "effect_mse": effect_sum / config["steps_per_epoch"],
            "expected_active": active_sum / config["steps_per_epoch"],
            "val_nrmse": val_score, "temperature": temperature,
        }
        curves.append(record)
        if epoch % 10 == 0 or epoch == config["max_epochs"] - 1:
            logger.log(
                f"epoch={epoch:03d} loss={record['loss']:.6f} effect={record['effect_mse']:.6f} "
                f"active={record['expected_active']:.3f} val_nrmse={val_score:.5f}"
            )
        if val_score < best_val - 1e-4:
            best_val, stale, best_epoch = val_score, 0, epoch
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        else:
            stale += 1
        torch.save(model.state_dict(), run_dir / "final_checkpoint.pt")
        dump_json(run_dir / "training_curves.json", curves)
        if stale >= config["patience"] and epoch >= 20:
            logger.log(f"early_stop epoch={epoch} best_epoch={best_epoch} best_val={best_val:.6f}")
            break
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", weights_only=True))
    metrics = evaluate(model, tp)
    metrics.update({
        "experiment": "O4", "tp": tp, "optimization_seed": seed,
        "best_epoch": best_epoch, "best_val_nrmse_subset": best_val,
        "epochs_run": len(curves), "wall_seconds": time.time() - started,
    })
    dump_json(metrics_path, metrics)
    logger.log(json.dumps(metrics, ensure_ascii=False))
    logger.close()
    return metrics


def summarize(tp: str) -> Dict[str, Any]:
    paths = sorted((OUTPUT_ROOT / "O4" / tp).glob("seed_*/metrics.json"))
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = aggregate_seed_metrics(metrics, {
        "iid_nrmse": ["prediction_nrmse", "test_iid"],
        "combination_101_nrmse": ["prediction_nrmse", "test_combination_101"],
        "iid_f1_mean": ["participation_f1_mean"],
        "iid_instance_r2_min": ["instance_effect_r2", "test_iid", "minimum_matched"],
        "functional_r2_min": ["functional_r2", "minimum_matched"],
        "iid_expected_active": ["expected_active"],
    })
    summary.update({"experiment": "O4", "tp": tp, "pass": len(metrics) == 3 and all(x["pass"] for x in metrics)})
    dump_json(OUTPUT_ROOT / "O4" / tp / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", choices=["identity", "orthogonal", "both"], default="both")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    conditions = ["identity", "orthogonal"] if args.tp == "both" else [args.tp]
    for tp in conditions:
        for seed in [int(value) for value in args.seeds.split(",")]:
            train(tp, seed, args.force)
        print(json.dumps(summarize(tp), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
