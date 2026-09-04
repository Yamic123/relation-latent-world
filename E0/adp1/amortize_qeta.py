"""Phase-II amortization for E0-ADP1: distill the exact solver's teacher
assignments ``(m, v)`` into the original ``q_eta(S, p)`` architecture.

The mechanism bank and teacher assignments are frozen; the student is trained
from random initialization with ``L_amort = L_m + L_v`` (BCE on support logits +
masked MSE on active v).  No Hard-Concrete sampling and no extra sparsity term.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from adp1._utils import DEVICE, dump_json, to_tensor

import run_e0 as base  # noqa: E402  (E0_ROOT on sys.path via _utils)


def _amort_loss(logits: torch.Tensor, v_hat: torch.Tensor, m_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    lm = F.binary_cross_entropy_with_logits(logits, m_t, reduction="mean")
    num = (m_t * (v_hat - v_t).square()).sum()
    den = m_t.sum() + 1e-9
    lv = num / den
    return lm + lv


@torch.no_grad()
def _student_output(q: nn.Module, S: np.ndarray, p: np.ndarray, batch_size: int = 2048) -> Dict[str, np.ndarray]:
    q.eval()
    logits_parts = []
    v_parts = []
    for start in range(0, len(S), batch_size):
        end = min(start + batch_size, len(S))
        logits, v, _ = q(to_tensor(S[start:end]), to_tensor(p[start:end]))
        logits_parts.append(logits.cpu().numpy())
        v_parts.append(v.cpu().numpy())
    return {"logits": np.concatenate(logits_parts, axis=0), "v": np.concatenate(v_parts, axis=0)}


def train_qeta(
    seed: int,
    tp: str,
    bank: nn.Module,
    S_train: np.ndarray,
    p_train: np.ndarray,
    teacher_m: np.ndarray,
    teacher_v: np.ndarray,
    S_val: np.ndarray,
    p_val: np.ndarray,
    teacher_m_val: np.ndarray,
    teacher_v_val: np.ndarray,
    cfg: Mapping[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    a_cfg = cfg["amortization"]
    lr = a_cfg["learning_rate"]
    wd = a_cfg["weight_decay"]
    bs = a_cfg["batch_size"]
    max_epochs = a_cfg["max_epochs"]
    patience = a_cfg["patience"]
    clip = a_cfg.get("gradient_clip", 1.0)

    # Fresh random student; bank stays frozen.
    base.seed_everything(seed + 40_001)
    q = base.MechanismAbstraction(cfg["m_max"], 64, 2, 4).to(DEVICE)
    optimizer = torch.optim.AdamW(q.parameters(), lr=lr, weight_decay=wd)

    m_t = to_tensor(teacher_m.astype(np.float32))
    v_t = to_tensor(teacher_v.astype(np.float32))
    m_val = to_tensor(teacher_m_val.astype(np.float32))
    v_val = to_tensor(teacher_v_val.astype(np.float32))
    S_val_t = to_tensor(S_val[:5000])
    p_val_t = to_tensor(p_val[:5000])

    rng = np.random.default_rng(seed + 40_002)
    best_val = float("inf")
    best_state = None
    stale = 0
    curves = []
    started = time.time()

    for epoch in range(max_epochs):
        q.train()
        loss_sum = 0.0
        for _ in range(64):
            idx = rng.integers(0, len(S_train), size=bs)
            logits, v_hat, _ = q(to_tensor(S_train[idx]), to_tensor(p_train[idx]))
            loss = _amort_loss(logits, v_hat, m_t[idx], v_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), clip)
            optimizer.step()
            loss_sum += float(loss.detach())

        q.eval()
        with torch.no_grad():
            logits_v, v_hat_v, _ = q(S_val_t, p_val_t)
            val_loss = float(_amort_loss(logits_v, v_hat_v, m_val[:5000], v_val[:5000]))
        curves.append({"epoch": epoch, "loss": loss_sum / 64, "val_amort_loss": val_loss})
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"  [amortize {tp}] seed={seed} epoch={epoch} loss={curves[-1]['loss']:.5f} val={val_loss:.5f}", flush=True)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in q.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience and epoch >= 20:
            break

    if best_state is not None:
        q.load_state_dict(best_state)
    torch.save(q.state_dict(), out_dir / "qeta_best.pt")
    dump_json(out_dir / "training_curves.json", curves)
    return {
        "tp": tp,
        "optimization_seed": seed,
        "epochs_run": len(curves),
        "best_val_amort_loss": best_val,
        "wall_seconds": time.time() - started,
    }


def teacher_fidelity(student_out: Dict[str, np.ndarray], teacher_m: np.ndarray, teacher_v: np.ndarray) -> Dict[str, float]:
    logits = student_out["logits"]
    m_hat = (torch.sigmoid(torch.from_numpy(logits)).numpy() >= 0.5).astype(np.float32)
    support_exact = float(np.mean(np.all(m_hat == teacher_m, axis=1)))
    f1s = []
    for j in range(teacher_m.shape[1]):
        tp = float(np.sum((m_hat[:, j] == 1) & (teacher_m[:, j] == 1)))
        fp = float(np.sum((m_hat[:, j] == 1) & (teacher_m[:, j] == 0)))
        fn = float(np.sum((m_hat[:, j] == 0) & (teacher_m[:, j] == 1)))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1e-12))
    active = teacher_m == 1
    denom = active.sum()
    mae = float(np.abs(student_out["v"] - teacher_v)[active].sum() / max(denom, 1e-9))
    ss_res = float(((student_out["v"] - teacher_v)[active] ** 2).sum())
    v_center = teacher_v[active]
    ss_tot = float(((v_center - v_center.mean()) ** 2).sum())
    v_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "teacher_support_f1_mean": float(np.mean(f1s)),
        "teacher_support_exact_accuracy": support_exact,
        "teacher_active_v_mae": mae,
        "teacher_active_v_r2": v_r2,
    }
