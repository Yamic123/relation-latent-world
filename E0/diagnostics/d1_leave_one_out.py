from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from common import (
    DEVICE,
    OUTPUT_ROOT,
    RunLogger,
    base,
    dump_config,
    dump_json,
    load_all,
    metadata,
    nrmse,
    seed_record,
)


def run(tp: str, seed: int, checkpoint: str | None = None) -> Dict[str, Any]:
    source = (
        base.OUTPUT_ROOT / "e0a" / "main" / tp / f"seed_{seed}" / "best_checkpoint.pt"
        if checkpoint is None else Path(checkpoint)
    )
    output = OUTPUT_ROOT / "D1" / tp / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "D1", "tp": tp, "source_checkpoint": str(source),
        "splits": ["test_iid", "test_context", "test_combination_101"],
        "damage_definition": "MSE(delta,prediction_without_l)-MSE(delta,full_prediction)",
        "frozen_original_failed_model": True,
    }
    dump_config(output / "config.yaml", config)
    dump_json(output / "seed.json", seed_record(seed))
    dump_json(output / "run_metadata.json", metadata("D1", seed, config))
    dump_json(output / "training_curves.json", [])
    logger = RunLogger(output / "stdout.log")
    logger.log(f"D1 tp={tp} seed={seed} checkpoint={source} device={DEVICE}")
    started = time.time()
    model = base.E0AModel(5, 64, 2, 4).to(DEVICE)
    model.load_state_dict(torch.load(source, map_location=DEVICE, weights_only=True))
    splits: Dict[str, Any] = {}
    for split in config["splits"]:
        data = load_all(split, tp)
        predicted = base.predict_a(model, data["S"], data["p"])
        target = data["delta_S"]
        full = predicted["prediction"]
        full_mse = float(np.mean((target - full) ** 2))
        candidates: Dict[str, Any] = {}
        for candidate in range(5):
            without = full - predicted["effects"][:, candidate]
            without_mse = float(np.mean((target - without) ** 2))
            norms = np.linalg.norm(predicted["effects"][:, candidate].reshape(len(target), -1), axis=1)
            candidates[f"candidate_{candidate + 1}"] = {
                "mean_gate": float(predicted["gate_probs"][:, candidate].mean()),
                "mean_effect_norm": float(norms.mean()),
                "full_mse": full_mse,
                "leave_one_out_mse": without_mse,
                "G_l": without_mse - full_mse,
            }
        splits[split] = {
            "full_nrmse": nrmse(target, full), "full_mse": full_mse,
            "candidates": candidates,
            "all_candidates_necessary_G_positive": all(x["G_l"] > 0 for x in candidates.values()),
        }
        logger.log(f"{split}: " + json.dumps(candidates, ensure_ascii=False))
    metrics = {
        "experiment": "D1", "tp": tp, "optimization_seed": seed,
        "splits": splits,
        "distributed_cooperative_code": all(
            item["all_candidates_necessary_G_positive"] for item in splits.values()
        ),
        "wall_seconds": time.time() - started,
    }
    dump_json(output / "metrics.json", metrics)
    logger.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", choices=["identity", "orthogonal"], default="identity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    print(json.dumps(run(args.tp, args.seed, args.checkpoint), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
