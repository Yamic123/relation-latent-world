from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
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
    dump_config,
    dump_json,
    instance_r2_matrix,
    load_all,
    metadata,
    nrmse,
    slices,
)


CONFIG: Dict[str, Any] = {
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "max_epochs": 200,
    "steps_per_epoch": 64,
    "gradient_clip": 1.0,
    "token_dim": 64,
    "attention_heads": 4,
    "M": 3,
    "validation_subset": 5000,
    "patience": 30,
}


class O1Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bank = base.SetMechanismBank(3, CONFIG["token_dim"], CONFIG["attention_heads"])

    def forward(self, S: torch.Tensor, m: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.bank.raw_effects(S, v)
        effects = m[:, :, None, None] * raw
        return {"prediction": effects.sum(dim=1), "effects": effects, "raw": raw}


@torch.no_grad()
def predict(model: O1Model, data: Dict[str, np.ndarray], batch_size: int = 2048) -> Dict[str, np.ndarray]:
    model.eval()
    predictions: List[np.ndarray] = []
    effects: List[np.ndarray] = []
    for sl in slices(len(data["S"]), batch_size):
        output = model(
            torch.from_numpy(data["S"][sl]).to(DEVICE),
            torch.from_numpy(data["m_gt"][sl]).to(DEVICE),
            torch.from_numpy(data["v_gt"][sl]).to(DEVICE),
        )
        predictions.append(output["prediction"].cpu().numpy())
        effects.append(output["effects"].cpu().numpy())
    return {"prediction": np.concatenate(predictions), "effects": np.concatenate(effects)}


def evaluate(model: O1Model) -> Dict[str, Any]:
    prediction_metrics: Dict[str, float] = {}
    instance_metrics: Dict[str, Any] = {}
    for split in base.SPLIT_SIZES:
        data = load_all(split)
        output = predict(model, data)
        prediction_metrics[split] = nrmse(data["delta_S"], output["prediction"])
        matrix = instance_r2_matrix(output["effects"], data["e_gt"])
        instance_metrics[split] = {
            "matrix": matrix.tolist(),
            "diagonal": np.diag(matrix).tolist(),
            "minimum_diagonal": float(np.diag(matrix).min()),
        }
    pass_components = {
        "iid_nrmse": prediction_metrics["test_iid"] < 0.05,
        "combination_101_nrmse": prediction_metrics["test_combination_101"] < 0.05,
    }
    return {
        "prediction_nrmse": prediction_metrics,
        "instance_effect_r2": instance_metrics,
        "pass_components": pass_components,
        "strong_pass": prediction_metrics["test_iid"] < 0.02 and prediction_metrics["test_combination_101"] < 0.02,
        "pass": bool(all(pass_components.values())),
    }


def train(seed: int, force: bool = False) -> Dict[str, Any]:
    run_dir = OUTPUT_ROOT / "O1" / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(run_dir / "config.yaml", CONFIG)
    dump_json(run_dir / "seed.json", {"world_seed": base.WORLD_SEED, "dataset_seed": base.DATASET_SEED, "optimization_seed": seed})
    dump_json(run_dir / "run_metadata.json", metadata("O1", seed, CONFIG))
    logger = RunLogger(run_dir / "stdout.log")
    base.seed_everything(seed)
    train_data = load_all("train")
    val_data = load_all("val")
    model = O1Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    rng = np.random.default_rng(seed + 1_010_101)
    curves: List[Dict[str, float]] = []
    best_val = float("inf")
    stale = 0
    started = time.time()
    logger.log(f"O1 seed={seed} device={DEVICE}")
    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        loss_sum = 0.0
        for _ in range(CONFIG["steps_per_epoch"]):
            idx = rng.integers(0, len(train_data["S"]), CONFIG["batch_size"])
            output = model(
                torch.from_numpy(train_data["S"][idx]).to(DEVICE),
                torch.from_numpy(train_data["m_gt"][idx]).to(DEVICE),
                torch.from_numpy(train_data["v_gt"][idx]).to(DEVICE),
            )
            loss = F.mse_loss(output["prediction"], torch.from_numpy(train_data["delta_S"][idx]).to(DEVICE))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach())
        subset = {key: value[: CONFIG["validation_subset"]] for key, value in val_data.items()}
        val_prediction = predict(model, subset)["prediction"]
        val_score = nrmse(subset["delta_S"], val_prediction)
        curves.append({"epoch": epoch, "train_mse": loss_sum / CONFIG["steps_per_epoch"], "val_nrmse": val_score})
        if epoch % 10 == 0 or epoch == CONFIG["max_epochs"] - 1:
            logger.log(f"epoch={epoch:03d} mse={curves[-1]['train_mse']:.7f} val_nrmse={val_score:.5f}")
        if val_score < best_val - 1e-5:
            best_val = val_score
            stale = 0
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        else:
            stale += 1
        dump_json(run_dir / "training_curves.json", curves)
        if stale >= CONFIG["patience"] and epoch >= 50:
            logger.log(f"early_stop epoch={epoch} best_val={best_val:.6f}")
            break
    torch.save(model.state_dict(), run_dir / "final_checkpoint.pt")
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", weights_only=True))
    metrics = evaluate(model)
    metrics.update({"experiment": "O1", "optimization_seed": seed, "best_val_nrmse_subset": best_val, "epochs_run": len(curves), "wall_seconds": time.time() - started})
    dump_json(metrics_path, metrics)
    logger.log(json.dumps(metrics, ensure_ascii=False))
    logger.close()
    return metrics


def summarize() -> Dict[str, Any]:
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((OUTPUT_ROOT / "O1").glob("seed_*/metrics.json"))]
    summary = aggregate_seed_metrics(
        metrics,
        {
            "iid_nrmse": ["prediction_nrmse", "test_iid"],
            "context_nrmse": ["prediction_nrmse", "test_context"],
            "combination_101_nrmse": ["prediction_nrmse", "test_combination_101"],
            "instance_r2_min": ["instance_effect_r2", "test_iid", "minimum_diagonal"],
        },
    )
    summary.update({"experiment": "O1", "pass": len(metrics) == 3 and all(item["pass"] for item in metrics)})
    dump_json(OUTPUT_ROOT / "O1" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for seed in [int(value) for value in args.seeds.split(",")]:
        train(seed, force=args.force)
    summarize()


if __name__ == "__main__":
    main()
