"""Residual / backfitting M-step for E0-ADP1.

Freezes the stored ``m``/``v`` assignments and, for each candidate in a seeded
order, fits ``M_j`` to the residual ``delta - stopgrad(sum_{k != j} m_k M_k)``.
The shared backbone is allowed to update, but the other candidates' adapter rows
are kept bit-identical via snapshot/restore of both their parameter values and
their AdamW moments (weight decay and moment leakage would otherwise move them).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from adp1._utils import DEVICE, mix_seed


def _probe_effects(bank: torch.nn.Module, probe: Optional["tuple[torch.Tensor, torch.Tensor]"]) -> Optional[np.ndarray]:
    if probe is None:
        return None
    S_probe, v_probe = probe
    with torch.no_grad():
        raw = bank.raw_effects(S_probe, v_probe)  # [P,5,3,2]
    return raw.cpu().numpy()


def _probe_drift(before: Optional[np.ndarray], after: Optional[np.ndarray]) -> Optional[List[float]]:
    if before is None or after is None:
        return None
    # Per-candidate mean effect change (L2 over probe slots/coords, then mean over probe).
    diff = after - before  # [P,5,3,2]
    per_candidate = np.sqrt((diff ** 2).sum(axis=(-2, -1))).mean(axis=0)  # [5]
    return per_candidate.tolist()


def m_step_round(
    bank: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    m: np.ndarray,
    v: np.ndarray,
    S_train: np.ndarray,
    delta_train: np.ndarray,
    cfg: Mapping[str, Any],
    round_id: int,
    seed: int,
    probe: Optional["tuple[torch.Tensor, torch.Tensor]"] = None,
    shared_before: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Run one EM round of the M-step over all five candidates."""
    m_cfg = cfg["m_step"]
    bs = m_cfg["batch_size"]
    steps = m_cfg["steps_per_candidate_per_round"]
    clip = m_cfg["gradient_clip"]

    order_rng = np.random.default_rng(mix_seed(seed, round_id))
    order = order_rng.permutation(5).tolist()
    sample_rng = np.random.default_rng(mix_seed(seed, round_id, 0xC0FFEE))

    S_t = torch.from_numpy(S_train).to(DEVICE)
    v_t = torch.from_numpy(v).to(DEVICE)
    d_t = torch.from_numpy(delta_train).to(DEVICE)
    m_t = torch.from_numpy(m.astype(np.float32)).to(DEVICE)

    adapters = bank.adapters
    probe_before = _probe_effects(bank, probe)

    records: List[Dict[str, Any]] = []
    for j in order:
        active_ids = np.flatnonzero(m[:, j].astype(bool))
        if active_ids.size == 0:
            records.append({"candidate": j, "skipped_empty_candidate": True})
            continue

        # Snapshot candidate-j-excluded adapter rows (value + Adam moments).
        adp_val = adapters.detach().clone()
        if adapters in optimizer.state:
            m_snap = optimizer.state[adapters]["exp_avg"].clone()
            v_snap = optimizer.state[adapters]["exp_avg_sq"].clone()
        else:
            m_snap = torch.zeros_like(adapters)
            v_snap = torch.zeros_like(adapters)
        keep = torch.ones(5, dtype=torch.bool, device=adapters.device)
        keep[j] = False

        for _ in range(steps):
            idx = sample_rng.integers(0, active_ids.size, size=bs)
            batch = active_ids[idx]
            raw = bank.raw_effects(S_t[batch], v_t[batch])  # [bs,5,3,2]
            pred_j = raw[:, j]
            other = torch.zeros_like(raw[:, 0])
            for k in range(5):
                if k != j:
                    other = other + m_t[batch, k, None, None] * raw[:, k].detach()
            residual = d_t[batch] - other
            loss = F.mse_loss(pred_j, residual)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bank.parameters(), clip)
            optimizer.step()
            # Restore non-j adapter rows (undo weight decay + moment leakage).
            with torch.no_grad():
                adapters.data[keep] = adp_val[keep]
                st = optimizer.state[adapters]
                st["exp_avg"][keep] = m_snap[keep]
                st["exp_avg_sq"][keep] = v_snap[keep]

        records.append({"candidate": j, "skipped_empty_candidate": False, "n_active": int(active_ids.size)})

    probe_after = _probe_effects(bank, probe)
    shared_drift_norm = None
    if shared_before is not None:
        total = 0.0
        for name, param in bank.named_parameters():
            if name == "adapters":
                continue
            total += float(torch.sum((param.detach() - shared_before[name].detach()) ** 2))
        shared_drift_norm = float(total ** 0.5)

    return {
        "order": order,
        "candidates": records,
        "shared_drift_norm": shared_drift_norm,
        "probe_effect_drift": _probe_drift(probe_before, probe_after),
    }


def snapshot_shared(bank: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in bank.named_parameters() if name != "adapters"}


def make_probe(n: int = 512, seed: int = 20260901) -> "tuple[torch.Tensor, torch.Tensor]":
    rng = np.random.default_rng(seed)
    S_probe = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(n, 3, 2)).astype(np.float32)).to(DEVICE)
    v_probe = torch.from_numpy(rng.uniform(-1.5, 1.5, size=(n, 5)).astype(np.float32)).to(DEVICE)
    return S_probe, v_probe
