from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

import numpy as np
import torch
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
    gt_effects_torch,
    instance_r2_matrix,
    load_all,
    metadata,
    nrmse,
    r2,
    seed_record,
    slices,
    world_tensors,
)


BASE_CONFIG: Dict[str, Any] = {
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "max_epochs": 200,
    "steps_per_epoch": 64,
    "patience": 20,
    "gradient_clip": 1.0,
    "lambda_participation": 1e-3,
    "temperature_start": 1.0,
    "temperature_end": 0.35,
    "token_dim": 64,
    "attention_heads": 4,
    "cross_attention_layers": 2,
    "M": 3,
    "validation_subset": 5000,
    "mechanisms": "fixed exact ground truth",
    "training_contract": "visible inputs/targets only; hidden labels used only in evaluation",
}


class O3AModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.abstraction = base.MechanismAbstraction(
            3, BASE_CONFIG["token_dim"], BASE_CONFIG["cross_attention_layers"],
            BASE_CONFIG["attention_heads"],
        )
        fixed = world_tensors()
        for key, value in fixed.items():
            self.register_buffer(f"world_{key}", value)

    def world(self) -> Dict[str, torch.Tensor]:
        return {key: getattr(self, f"world_{key}") for key in ["C", "A", "b", "c"]}

    def forward(self, S: torch.Tensor, p: torch.Tensor, temperature: float) -> Dict[str, torch.Tensor]:
        logits, v, _ = self.abstraction(S, p)
        gates, probs = base.hard_concrete(logits, self.training, temperature)
        effects = gt_effects_torch(S, gates, v, self.world())
        return {
            "prediction": effects.sum(dim=1), "effects": effects, "gates": gates,
            "gate_probs": probs, "v": v, "logits": logits,
        }


@torch.no_grad()
def predict(model: O3AModel, data: Dict[str, np.ndarray], batch_size: int = 2048) -> Dict[str, np.ndarray]:
    model.eval()
    parts: Dict[str, List[np.ndarray]] = {
        key: [] for key in ["prediction", "effects", "gate_probs", "v", "logits"]
    }
    for sl in slices(len(data["S"]), batch_size):
        result = model(
            torch.from_numpy(data["S"][sl]).to(DEVICE),
            torch.from_numpy(data["p"][sl]).to(DEVICE),
            0.35,
        )
        for key in parts:
            parts[key].append(result[key].cpu().numpy())
    return {key: np.concatenate(value) for key, value in parts.items()}


def evaluate_split(model: O3AModel, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    output = predict(model, data)
    participation: Dict[str, Any] = {}
    realization: Dict[str, Any] = {}
    for j in range(3):
        participation[f"Z{j + 1}"] = binary_metrics(data["m_gt"][:, j], output["gate_probs"][:, j])
        active = data["m_gt"][:, j].astype(bool)
        if active.sum() >= 2:
            realization[f"Z{j + 1}"] = {
                "r2": r2(data["v_gt"][active, j, None], output["v"][active, j, None]),
                "mae": float(np.mean(np.abs(data["v_gt"][active, j] - output["v"][active, j]))),
            }
        else:
            realization[f"Z{j + 1}"] = {"r2": None, "mae": None}
    matrix = instance_r2_matrix(output["effects"], data["e_gt"])
    f1s = [entry["f1"] for entry in participation.values()]
    aucs = [entry["auroc"] for entry in participation.values()]
    diagonal = np.diag(matrix)
    return {
        "prediction_nrmse": nrmse(data["delta_S"], output["prediction"]),
        "participation": participation,
        "participation_f1_mean": float(np.mean(f1s)),
        "participation_f1_min": float(np.min(f1s)),
        "participation_auroc_mean": float(np.mean(aucs)),
        "participation_auroc_min": float(np.min(aucs)),
        "realization": realization,
        "instance_effect_r2": {
            "matrix": matrix.tolist(), "diagonal": diagonal.tolist(),
            "minimum_diagonal": float(diagonal.min()),
        },
        "expected_active": float(output["gate_probs"].sum(axis=1).mean()),
        "candidate_usage": output["gate_probs"].mean(axis=0).tolist(),
        "v_mean": output["v"].mean(axis=0).tolist(),
        "v_std": output["v"].std(axis=0).tolist(),
    }


def evaluate(model: O3AModel, tp: str) -> Dict[str, Any]:
    splits = {split: evaluate_split(model, load_all(split, tp)) for split in base.SPLIT_SIZES}
    iid = splits["test_iid"]
    passes = {
        "iid_nrmse_lt_0.05": iid["prediction_nrmse"] < 0.05,
        "iid_f1_min_gt_0.9": iid["participation_f1_min"] > 0.9,
    }
    return {"splits": splits, "pass_components": passes, "pass": bool(all(passes.values()))}


def train(tp: str, seed: int, force: bool = False) -> Dict[str, Any]:
    config = {**BASE_CONFIG, "tp": tp}
    run_dir = OUTPUT_ROOT / "O3A" / tp / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(run_dir / "config.yaml", config)
    dump_json(run_dir / "seed.json", seed_record(seed))
    dump_json(run_dir / "run_metadata.json", metadata("O3-A", seed, config))
    logger = RunLogger(run_dir / "stdout.log")
    base.seed_everything(seed)
    train_data = load_all("train", tp)
    val_data = load_all("val", tp)
    model = O3AModel().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.abstraction.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    rng = np.random.default_rng(seed + (3_010_000 if tp == "identity" else 3_020_000))
    curves: List[Dict[str, float]] = []
    best_val, stale, best_epoch = float("inf"), 0, -1
    started = time.time()
    logger.log(f"O3-A tp={tp} seed={seed} device={DEVICE}")
    for epoch in range(config["max_epochs"]):
        model.train()
        fraction = epoch / max(config["max_epochs"] - 1, 1)
        temperature = config["temperature_start"] * (
            config["temperature_end"] / config["temperature_start"]
        ) ** fraction
        total_sum = effect_sum = active_sum = 0.0
        for _ in range(config["steps_per_epoch"]):
            idx = rng.integers(0, len(train_data["S"]), config["batch_size"])
            result = model(
                torch.from_numpy(train_data["S"][idx]).to(DEVICE),
                torch.from_numpy(train_data["p"][idx]).to(DEVICE),
                temperature,
            )
            target = torch.from_numpy(train_data["delta_S"][idx]).to(DEVICE)
            effect_loss = F.mse_loss(result["prediction"], target)
            expected_active = result["gate_probs"].sum(dim=1).mean()
            loss = effect_loss + config["lambda_participation"] * expected_active
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.abstraction.parameters(), config["gradient_clip"])
            optimizer.step()
            total_sum += float(loss.detach()); effect_sum += float(effect_loss.detach()); active_sum += float(expected_active.detach())
        subset = {key: value[: config["validation_subset"]] for key, value in val_data.items()}
        validation = evaluate_split(model, subset)
        score = validation["prediction_nrmse"]
        record = {
            "epoch": epoch, "loss": total_sum / config["steps_per_epoch"],
            "effect_mse": effect_sum / config["steps_per_epoch"],
            "expected_active": active_sum / config["steps_per_epoch"],
            "val_nrmse": score, "temperature": temperature,
        }
        curves.append(record)
        if epoch % 10 == 0 or epoch == config["max_epochs"] - 1:
            logger.log(
                f"epoch={epoch:03d} loss={record['loss']:.6f} effect={record['effect_mse']:.6f} "
                f"active={record['expected_active']:.3f} val_nrmse={score:.5f}"
            )
        if score < best_val - 1e-4:
            best_val, stale, best_epoch = score, 0, epoch
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
        "experiment": "O3-A", "tp": tp, "optimization_seed": seed,
        "best_epoch": best_epoch, "best_val_nrmse_subset": best_val,
        "epochs_run": len(curves), "wall_seconds": time.time() - started,
    })
    dump_json(metrics_path, metrics)
    logger.log(json.dumps(metrics, ensure_ascii=False))
    logger.close()
    return metrics


def summarize(tp: str) -> Dict[str, Any]:
    paths = sorted((OUTPUT_ROOT / "O3A" / tp).glob("seed_*/metrics.json"))
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = aggregate_seed_metrics(metrics, {
        "iid_nrmse": ["splits", "test_iid", "prediction_nrmse"],
        "combination_101_nrmse": ["splits", "test_combination_101", "prediction_nrmse"],
        "iid_f1_min": ["splits", "test_iid", "participation_f1_min"],
        "iid_auroc_min": ["splits", "test_iid", "participation_auroc_min"],
        "iid_instance_r2_min": ["splits", "test_iid", "instance_effect_r2", "minimum_diagonal"],
        "iid_expected_active": ["splits", "test_iid", "expected_active"],
    })
    summary.update({"experiment": "O3-A", "tp": tp, "pass": len(metrics) == 3 and all(x["pass"] for x in metrics)})
    dump_json(OUTPUT_ROOT / "O3A" / tp / "summary.json", summary)
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
            train(tp, seed, force=args.force)
        print(json.dumps(summarize(tp), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
