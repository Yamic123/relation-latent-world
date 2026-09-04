from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
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
    load_all,
    metadata,
    r2,
    seed_record,
)


class Probe(nn.Module):
    def __init__(self, nonlinear: bool) -> None:
        super().__init__()
        self.network = (
            nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
            if nonlinear else nn.Linear(5, 3)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def extract_latent(tp: str, seed: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str]:
    source = base.OUTPUT_ROOT / "e0a" / "main" / tp / f"seed_{seed}" / "best_checkpoint.pt"
    model = base.E0AModel(5, 64, 2, 4).to(DEVICE)
    model.load_state_dict(torch.load(source, map_location=DEVICE, weights_only=True))
    result: List[Dict[str, np.ndarray]] = []
    for split in ["train", "test_iid"]:
        data = load_all(split, tp)
        prediction = base.predict_a(model, data["S"], data["p"])
        result.append({"x": prediction["v"], "m": data["m_gt"], "v": data["v_gt"]})
    return result[0], result[1], str(source)


def participation_metrics(probe: Probe, x: np.ndarray, m: np.ndarray) -> Dict[str, Any]:
    probe.eval()
    with torch.no_grad():
        score = torch.sigmoid(probe(torch.from_numpy(x).to(DEVICE))).cpu().numpy()
    per = {f"Z{j + 1}": binary_metrics(m[:, j], score[:, j]) for j in range(3)}
    return {
        "per_mechanism": per,
        "f1_mean": float(np.mean([value["f1"] for value in per.values()])),
        "f1_min": float(np.min([value["f1"] for value in per.values()])),
        "auroc_mean": float(np.mean([value["auroc"] for value in per.values()])),
        "auroc_min": float(np.min([value["auroc"] for value in per.values()])),
    }


def realization_metrics(probe: Probe, x: np.ndarray, m: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
    probe.eval()
    with torch.no_grad():
        prediction = probe(torch.from_numpy(x).to(DEVICE)).cpu().numpy()
    per: Dict[str, Any] = {}
    for j in range(3):
        active = m[:, j].astype(bool)
        per[f"Z{j + 1}"] = {
            "r2": r2(v[active, j, None], prediction[active, j, None]),
            "mae": float(np.mean(np.abs(v[active, j] - prediction[active, j]))),
            "n_active": int(active.sum()),
        }
    return {
        "per_mechanism": per,
        "r2_mean": float(np.mean([value["r2"] for value in per.values()])),
        "r2_min": float(np.min([value["r2"] for value in per.values()])),
        "mae_mean": float(np.mean([value["mae"] for value in per.values()])),
    }


def train_probe(
    task: str,
    nonlinear: bool,
    train: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    rng: np.random.Generator,
    logger: RunLogger,
    steps: int,
) -> Tuple[Probe, List[Dict[str, float]], Dict[str, Any]]:
    probe = Probe(nonlinear).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    curves: List[Dict[str, float]] = []
    for step in range(steps):
        idx = rng.integers(0, len(train["x"]), 1024)
        x = torch.from_numpy(train["x"][idx]).to(DEVICE)
        output = probe(x)
        if task == "participation":
            target = torch.from_numpy(train["m"][idx]).to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(output, target)
        else:
            mask = torch.from_numpy(train["m"][idx]).to(DEVICE)
            target = torch.from_numpy(train["v"][idx]).to(DEVICE)
            loss = ((output - target).square() * mask).sum() / mask.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            record = {"step": step, "loss": float(loss.detach())}
            curves.append(record)
            logger.log(f"probe={task}/{('mlp' if nonlinear else 'linear')} step={step:04d} loss={record['loss']:.6f}")
    metrics = (
        participation_metrics(probe, test["x"], test["m"])
        if task == "participation"
        else realization_metrics(probe, test["x"], test["m"], test["v"])
    )
    return probe, curves, metrics


def run(tp: str, seed: int, steps: int) -> Dict[str, Any]:
    output = OUTPUT_ROOT / "D2" / tp / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "D2", "tp": tp, "probe_input": "frozen failed-model v_hat[1:5] only",
        "probe_types": ["linear", "two_layer_mlp"], "tasks": ["GT participation", "GT active realization"],
        "optimizer": "AdamW", "learning_rate": 1e-3, "batch_size": 1024,
        "steps": steps, "frozen_original_model": True,
    }
    dump_config(output / "config.yaml", config)
    dump_json(output / "seed.json", seed_record(seed))
    dump_json(output / "run_metadata.json", metadata("D2", seed, config))
    logger = RunLogger(output / "stdout.log")
    base.seed_everything(seed + 5_000_000)
    train, test, source = extract_latent(tp, seed)
    mean, std = train["x"].mean(axis=0), train["x"].std(axis=0).clip(min=1e-6)
    train["x"] = ((train["x"] - mean) / std).astype(np.float32)
    test["x"] = ((test["x"] - mean) / std).astype(np.float32)
    logger.log(f"D2 tp={tp} seed={seed} source={source} device={DEVICE}")
    started = time.time()
    rng = np.random.default_rng(seed + 5_010_000)
    all_curves: Dict[str, Any] = {}
    all_metrics: Dict[str, Any] = {}
    states: Dict[str, Any] = {"normalization_mean": mean, "normalization_std": std}
    for task in ["participation", "realization"]:
        for nonlinear in [False, True]:
            name = f"{task}_{'mlp' if nonlinear else 'linear'}"
            probe, curves, result = train_probe(task, nonlinear, train, test, rng, logger, steps)
            all_curves[name] = curves
            all_metrics[name] = result
            states[name] = probe.state_dict()
    torch.save(states, output / "checkpoint.pt")
    dump_json(output / "training_curves.json", all_curves)
    metrics = {
        "experiment": "D2", "tp": tp, "optimization_seed": seed,
        "source_checkpoint": source, "test_iid": all_metrics,
        "interpretation": {
            "linear_high": bool(
                all_metrics["participation_linear"]["f1_mean"] > 0.9
                and all_metrics["realization_linear"]["r2_mean"] > 0.9
            ),
            "mlp_high": bool(
                all_metrics["participation_mlp"]["f1_mean"] > 0.9
                and all_metrics["realization_mlp"]["r2_mean"] > 0.9
            ),
        },
        "wall_seconds": time.time() - started,
    }
    dump_json(output / "metrics.json", metrics)
    logger.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", choices=["identity", "orthogonal"], default="identity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args.tp, args.seed, args.steps), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
