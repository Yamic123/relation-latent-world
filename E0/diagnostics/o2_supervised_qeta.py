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
    load_all,
    metadata,
    r2,
    slices,
)


BASE_CONFIG: Dict[str, Any] = {
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "max_epochs": 100,
    "steps_per_epoch": 64,
    "gradient_clip": 1.0,
    "lambda_v": 1.0,
    "token_dim": 64,
    "attention_heads": 4,
    "cross_attention_layers": 2,
    "M": 3,
    "validation_subset": 5000,
    "patience": 20,
}


@torch.no_grad()
def predict(model: base.MechanismAbstraction, data: Dict[str, np.ndarray], batch_size: int = 2048) -> Dict[str, np.ndarray]:
    model.eval()
    logits_parts: List[np.ndarray] = []
    v_parts: List[np.ndarray] = []
    for sl in slices(len(data["S"]), batch_size):
        logits, v, _ = model(
            torch.from_numpy(data["S"][sl]).to(DEVICE),
            torch.from_numpy(data["p"][sl]).to(DEVICE),
        )
        logits_parts.append(logits.cpu().numpy())
        v_parts.append(v.cpu().numpy())
    logits = np.concatenate(logits_parts)
    return {"logits": logits, "m_probability": 1.0 / (1.0 + np.exp(-logits)), "v": np.concatenate(v_parts)}


def split_metrics(output: Dict[str, np.ndarray], data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    participation: Dict[str, Any] = {}
    realization: Dict[str, Any] = {}
    for j in range(3):
        participation[f"Z{j + 1}"] = binary_metrics(data["m_gt"][:, j], output["m_probability"][:, j])
        active = data["m_gt"][:, j].astype(bool)
        if active.sum() >= 2:
            target = data["v_gt"][active, j]
            predicted = output["v"][active, j]
            realization[f"Z{j + 1}"] = {
                "r2": r2(target[:, None], predicted[:, None]),
                "mae": float(np.mean(np.abs(target - predicted))),
                "n_active": int(active.sum()),
            }
        else:
            realization[f"Z{j + 1}"] = {"r2": None, "mae": None, "n_active": int(active.sum())}
    f1_values = [value["f1"] for value in participation.values()]
    auc_values = [value["auroc"] for value in participation.values()]
    r2_values = [value["r2"] for value in realization.values() if value["r2"] is not None]
    mae_values = [value["mae"] for value in realization.values() if value["mae"] is not None]
    return {
        "participation": participation,
        "participation_f1_mean": float(np.mean(f1_values)),
        "participation_f1_min": float(np.min(f1_values)),
        "participation_auroc_mean": float(np.mean(auc_values)),
        "participation_auroc_min": float(np.min(auc_values)),
        "realization": realization,
        "realization_r2_mean": float(np.mean(r2_values)),
        "realization_r2_min": float(np.min(r2_values)),
        "realization_mae_mean": float(np.mean(mae_values)),
    }


def evaluate(model: base.MechanismAbstraction, tp: str) -> Dict[str, Any]:
    splits: Dict[str, Any] = {}
    for split in base.SPLIT_SIZES:
        data = load_all(split, tp)
        splits[split] = split_metrics(predict(model, data), data)
    iid = splits["test_iid"]
    pass_components = {
        "f1": iid["participation_f1_min"] > 0.95,
        "auroc": iid["participation_auroc_min"] > 0.98,
        "realization_r2": iid["realization_r2_min"] > 0.95,
    }
    return {"splits": splits, "pass_components": pass_components, "pass": bool(all(pass_components.values()))}


def supervised_loss(
    model: base.MechanismAbstraction,
    S: torch.Tensor,
    p: torch.Tensor,
    m_gt: torch.Tensor,
    v_gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, v, _ = model(S, p)
    m_loss = F.binary_cross_entropy_with_logits(logits, m_gt)
    v_loss = ((v - v_gt).square() * m_gt).sum() / m_gt.sum().clamp_min(1.0)
    return m_loss + BASE_CONFIG["lambda_v"] * v_loss, m_loss, v_loss


def train(tp: str, seed: int, force: bool = False) -> Dict[str, Any]:
    config = {**BASE_CONFIG, "tp": tp}
    run_dir = OUTPUT_ROOT / "O2" / tp / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(run_dir / "config.yaml", config)
    dump_json(run_dir / "seed.json", {"world_seed": base.WORLD_SEED, "dataset_seed": base.DATASET_SEED, "optimization_seed": seed})
    dump_json(run_dir / "run_metadata.json", metadata("O2", seed, config))
    logger = RunLogger(run_dir / "stdout.log")
    base.seed_everything(seed)
    train_data = load_all("train", tp)
    val_data = load_all("val", tp)
    model = base.MechanismAbstraction(3, config["token_dim"], config["cross_attention_layers"], config["attention_heads"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed + (2_010_000 if tp == "identity" else 2_020_000))
    curves: List[Dict[str, float]] = []
    best_val = float("inf")
    stale = 0
    started = time.time()
    logger.log(f"O2 tp={tp} seed={seed} device={DEVICE}")
    for epoch in range(config["max_epochs"]):
        model.train()
        total_sum = m_sum = v_sum = 0.0
        for _ in range(config["steps_per_epoch"]):
            idx = rng.integers(0, len(train_data["S"]), config["batch_size"])
            total, m_loss, v_loss = supervised_loss(
                model,
                torch.from_numpy(train_data["S"][idx]).to(DEVICE),
                torch.from_numpy(train_data["p"][idx]).to(DEVICE),
                torch.from_numpy(train_data["m_gt"][idx]).to(DEVICE),
                torch.from_numpy(train_data["v_gt"][idx]).to(DEVICE),
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            total_sum += float(total.detach()); m_sum += float(m_loss.detach()); v_sum += float(v_loss.detach())
        model.eval()
        with torch.no_grad():
            n_val = config["validation_subset"]
            val_total, val_m, val_v = supervised_loss(
                model,
                torch.from_numpy(val_data["S"][:n_val]).to(DEVICE),
                torch.from_numpy(val_data["p"][:n_val]).to(DEVICE),
                torch.from_numpy(val_data["m_gt"][:n_val]).to(DEVICE),
                torch.from_numpy(val_data["v_gt"][:n_val]).to(DEVICE),
            )
        record = {
            "epoch": epoch,
            "train_loss": total_sum / config["steps_per_epoch"],
            "train_m_loss": m_sum / config["steps_per_epoch"],
            "train_v_loss": v_sum / config["steps_per_epoch"],
            "val_loss": float(val_total),
            "val_m_loss": float(val_m),
            "val_v_loss": float(val_v),
        }
        curves.append(record)
        if epoch % 10 == 0 or epoch == config["max_epochs"] - 1:
            logger.log(f"epoch={epoch:03d} train={record['train_loss']:.6f} val={record['val_loss']:.6f} val_m={record['val_m_loss']:.6f} val_v={record['val_v_loss']:.6f}")
        if record["val_loss"] < best_val - 1e-5:
            best_val = record["val_loss"]
            stale = 0
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        else:
            stale += 1
        dump_json(run_dir / "training_curves.json", curves)
        if stale >= config["patience"] and epoch >= 40:
            logger.log(f"early_stop epoch={epoch} best_val={best_val:.6f}")
            break
    torch.save(model.state_dict(), run_dir / "final_checkpoint.pt")
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", weights_only=True))
    metrics = evaluate(model, tp)
    metrics.update({"experiment": "O2", "tp": tp, "optimization_seed": seed, "best_val_loss": best_val, "epochs_run": len(curves), "wall_seconds": time.time() - started})
    dump_json(metrics_path, metrics)
    logger.log(json.dumps(metrics, ensure_ascii=False))
    logger.close()
    return metrics


def summarize(tp: str) -> Dict[str, Any]:
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((OUTPUT_ROOT / "O2" / tp).glob("seed_*/metrics.json"))]
    summary = aggregate_seed_metrics(
        metrics,
        {
            "iid_f1_min": ["splits", "test_iid", "participation_f1_min"],
            "iid_auroc_min": ["splits", "test_iid", "participation_auroc_min"],
            "iid_realization_r2_min": ["splits", "test_iid", "realization_r2_min"],
            "iid_realization_mae": ["splits", "test_iid", "realization_mae_mean"],
        },
    )
    summary.update({"experiment": "O2", "tp": tp, "pass": len(metrics) == 3 and all(item["pass"] for item in metrics)})
    dump_json(OUTPUT_ROOT / "O2" / tp / "summary.json", summary)
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
