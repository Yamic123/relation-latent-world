"""Small, non-destructive ADP12 integration smoke test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    from . import run_adp12 as m
except ImportError:  # Direct execution: python E0/adp12/smoke_adp12.py
    import run_adp12 as m


def take(x, n):
    return {k: v[:n] if isinstance(v, np.ndarray) and len(v) == len(x["S"]) else v for k, v in x.items()}


def main():
    c = m.cfg(997)
    c.update({"epochs_per_round": 1, "family_batch_size": 4, "death_bootstrap_repeats": 3})
    train = take(m.adp6.response_data("train"), 8)
    val = take(m.adp6.response_data("val"), 12)
    alpha_train, _ = m.adp6.coordinates("train")
    alpha_val, alpha_held = m.adp6.coordinates("val")
    alpha_train, alpha_val, alpha_held = alpha_train[:8], alpha_val[:12], alpha_held[:12]

    bank, a, b, q = m.init(997, c, 8)
    opt = torch.optim.AdamW(list(bank.parameters()) + [a, b], lr=1e-4)
    loss, _ = m.weighted_train(bank, a, b, train["S"], train["response_train"], alpha_train, q, c, opt, 0)
    cost = m.adp6.family_e_step_structured(bank, a, b, train["S"], train["response_train"], alpha_train)[1]
    soft, gap, tau = m.soft_q(cost, list(range(5)), 4.0, 1e-8)
    hard = m.hard_q(cost, list(range(5)))
    assert np.allclose(soft.sum(1), 1) and np.allclose(hard.sum(1), 1)

    val_cost = m.adp6.family_e_step_structured(bank, a, b, val["S"], val["response_train"], alpha_val)[1]
    vq, _, _ = m.soft_q(val_cost, list(range(5)), 4.0, 1e-8)
    death = m.death_test(4, vq, bank, a, b, val, alpha_val, alpha_held, c, 0)

    b1c = dict(c)
    b1c["b1"] = dict(c["b1"], steps=1, restarts=1, chunk_size=4)
    S, _, target = m.base.load_visible("e0a", "test_iid", "identity")
    latent = m.adp9.infer_support(bank, a, b, [0, 1, 2], S[:4], target[:4], b1c, "adp12_smoke")
    assert latent["prediction"].shape == target[:4].shape

    result = {
        "pass": True,
        "device": str(m.DEVICE),
        "weighted_loss": loss,
        "cost_shape": list(cost.shape),
        "soft_row_sum_max_error": float(np.abs(soft.sum(1) - 1).max()),
        "hard_row_sum_max_error": float(np.abs(hard.sum(1) - 1).max()),
        "median_gap": gap,
        "temperature": tau,
        "death_test_checks": death["checks"],
        "b1_prediction_shape": list(latent["prediction"].shape),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if m.DEVICE.type == "cuda" else 0.0,
    }
    out = m.OUT / "_smoke" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
