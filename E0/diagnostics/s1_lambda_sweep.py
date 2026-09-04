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


LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def lambda_name(value: float) -> str:
    return f"lambda_{value:.0e}".replace("+", "")


def evaluate(model: base.E0AModel) -> Dict[str, Any]:
    split_results: Dict[str, Any] = {}
    for split in ["test_iid", "test_context", "test_combination_101"]:
        data = load_all(split, "identity")
        output = base.predict_a(model, data["S"], data["p"])
        matrix = instance_r2_matrix(output["effects"], data["e_gt"])
        rows, cols = linear_sum_assignment(-np.nan_to_num(matrix, nan=-1e6))
        mapping = {int(gt): int(learned) for learned, gt in zip(rows, cols)}
        matched = [float(matrix[mapping[gt], gt]) for gt in range(3)]
        participation = {
            f"Z{gt + 1}": binary_metrics(data["m_gt"][:, gt], output["gate_probs"][:, mapping[gt]])
            for gt in range(3)
        }
        split_results[split] = {
            "prediction_nrmse": nrmse(data["delta_S"], output["prediction"]),
            "expected_active": float(output["gate_probs"].sum(axis=1).mean()),
            "candidate_usage": output["gate_probs"].mean(axis=0).tolist(),
            "participation_f1_mean": float(np.mean([x["f1"] for x in participation.values()])),
            "participation_f1_min": float(np.min([x["f1"] for x in participation.values()])),
            "participation": participation,
            "instance_effect_r2_matrix": matrix.tolist(),
            "instance_mapping_gt_to_learned": mapping,
            "instance_effect_r2_matched": matched,
            "instance_effect_r2_min": float(min(matched)),
        }
    iid = split_results["test_iid"]
    good_working_point = bool(
        iid["prediction_nrmse"] < 0.05
        and iid["participation_f1_mean"] > 0.9
        and iid["instance_effect_r2_min"] > 0.9
    )
    return {"splits": split_results, "good_working_point": good_working_point}


def train(lambda_p: float, seed: int, force: bool = False) -> Dict[str, Any]:
    run_dir = OUTPUT_ROOT / "S1" / lambda_name(lambda_p) / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    config: Dict[str, Any] = {
        "experiment": "S1", "tp": "identity", "m_max": 5,
        "lambda_participation": lambda_p, "optimizer": "AdamW", "learning_rate": 3e-4,
        "weight_decay": 1e-5, "batch_size": 512, "max_epochs": 200,
        "steps_per_epoch": 64, "patience": 20, "gradient_clip": 1.0,
        "temperature_start": 1.0, "temperature_end": 0.35,
        "token_dim": 64, "attention_heads": 4, "cross_attention_layers": 2,
        "validation_subset": 5000,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(run_dir / "config.yaml", config)
    dump_json(run_dir / "seed.json", seed_record(seed))
    dump_json(run_dir / "run_metadata.json", metadata("S1", seed, config))
    logger = RunLogger(run_dir / "stdout.log")
    base.seed_everything(seed)
    train_data = load_all("train", "identity")
    val_data = load_all("val", "identity")
    model = base.E0AModel(5, 64, 2, 4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed + int(round(-np.log10(lambda_p))) * 100_000 + 6_000_000)
    curves: List[Dict[str, float]] = []
    best_val, stale, best_epoch = float("inf"), 0, -1
    started = time.time()
    logger.log(f"S1 lambda={lambda_p:g} seed={seed} device={DEVICE}")
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
            active = output["gate_probs"].sum(dim=1).mean()
            loss = effect_loss + lambda_p * active
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach()); effect_sum += float(effect_loss.detach()); active_sum += float(active.detach())
        model.eval()
        val_output = base.predict_a(
            model, val_data["S"][: config["validation_subset"]], val_data["p"][: config["validation_subset"]]
        )
        val_score = nrmse(val_data["delta_S"][: config["validation_subset"]], val_output["prediction"])
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
    metrics = evaluate(model)
    metrics.update({
        "experiment": "S1", "lambda_participation": lambda_p, "optimization_seed": seed,
        "best_epoch": best_epoch, "best_val_nrmse_subset": best_val,
        "epochs_run": len(curves), "wall_seconds": time.time() - started,
    })
    dump_json(metrics_path, metrics)
    logger.log(json.dumps(metrics, ensure_ascii=False))
    logger.close()
    return metrics


def summarize(seed: int) -> Dict[str, Any]:
    runs = []
    for value in LAMBDAS:
        path = OUTPUT_ROOT / "S1" / lambda_name(value) / f"seed_{seed}" / "metrics.json"
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    table = []
    for item in runs:
        iid = item["splits"]["test_iid"]
        table.append({
            "lambda_participation": item["lambda_participation"],
            "iid_nrmse": iid["prediction_nrmse"],
            "combination_101_nrmse": item["splits"]["test_combination_101"]["prediction_nrmse"],
            "active_count": iid["expected_active"], "f1_mean": iid["participation_f1_mean"],
            "instance_r2_min": iid["instance_effect_r2_min"],
            "good_working_point": item["good_working_point"],
        })
    summary = {"experiment": "S1", "seed": seed, "table": table,
               "any_good_working_point": any(x["good_working_point"] for x in table)}
    dump_json(OUTPUT_ROOT / "S1" / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambdas", default=",".join(str(x) for x in LAMBDAS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    values = [float(x) for x in args.lambdas.split(",")]
    for value in values:
        train(value, args.seed, args.force)
    if values == LAMBDAS:
        print(json.dumps(summarize(args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
