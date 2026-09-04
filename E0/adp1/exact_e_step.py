"""Exact E-step for E0-ADP1: enumerate all 32 binary supports and optimize the
active continuous ``v`` per (sample, support, restart) with a batched Adam
solver over ``v = 1.5 * tanh(z)``.

The engineering vectorization batches ``sample x restart`` (and chunks samples)
on the GPU but preserves the exact semantics required by the guide: 32 supports,
5 restarts, the ``L_effect + 1e-3 |m|`` objective, the ``[-1.5, 1.5]`` bound via
``tanh``, per-sample support selection with the fixed tie-break, and per-instance
early stopping (relative tolerance 1e-7, patience 10).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from adp1._utils import (
    DEVICE,
    LAMBDA_P,
    NUM_SUPPORTS,
    SUPPORTS,
    SUPPORT_SIZES,
    mix_seed,
    tie_break_argmin,
)

# Full fp32 matmuls for numerical fidelity: the continuous-solver sanity check
# compares Adam against an L-BFGS-B reference to 1e-5, which TF32 (10-bit
# mantissa) would perturb at the ~1e-3 level.  This flag is recorded in metadata.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def freeze_bank(bank: torch.nn.Module) -> None:
    bank.eval()
    for p in bank.parameters():
        p.requires_grad_(False)


def unfreeze_bank(bank: torch.nn.Module) -> None:
    for p in bank.parameters():
        p.requires_grad_(True)
    bank.train()


def _adam_active_v(
    bank: torch.nn.Module,
    S_t: torch.Tensor,
    d_t: torch.Tensor,
    active_idx: List[int],
    z_inits: torch.Tensor,
    cfg: Mapping[str, Any],
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Optimize active v for a chunk of instances with 5 restarts.

    ``S_t``/``d_t`` are [B, 3, 2] where B = chunk_size * R (R restarts, sample
    major).  ``z_inits`` is [B, a].  Returns ``(best_z [B, a], best_loss [B])``.
    """
    B, a = z_inits.shape
    if a == 0:
        return z_inits, torch.zeros(B, device=z_inits.device)
    z = z_inits.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=cfg["learning_rate"])

    support_mask = torch.zeros(5, device=z.device)
    support_mask[active_idx] = 1.0
    support_mask = support_mask[None, :, None, None]  # [1,5,1,1]

    best_loss = torch.full((B,), 1e30, device=z.device)
    best_z = z.detach().clone()
    stale = torch.zeros(B, device=z.device)
    active_flag = torch.ones(B, device=z.device, dtype=torch.bool)

    rel_tol = cfg["relative_tolerance"]
    patience = cfg["tolerance_patience"]
    max_steps = cfg["max_steps"]

    steps_run = 0
    for _ in range(max_steps):
        steps_run += 1
        opt.zero_grad(set_to_none=True)
        v = 1.5 * torch.tanh(z)
        v_full = torch.zeros(B, 5, device=z.device)
        v_full[:, active_idx] = v
        raw = bank.raw_effects(S_t, v_full)  # [B,5,3,2]
        pred = (raw * support_mask).sum(dim=1)  # [B,3,2]
        lp = ((pred - d_t).square()).mean(dim=(-2, -1))  # [B]

        lpd = lp.detach()
        threshold = rel_tol * best_loss.abs().clamp_min(1e-12)
        improved = lpd < best_loss - threshold
        upd = improved & active_flag
        best_loss = torch.where(upd, lpd, best_loss)
        best_z = torch.where(upd[:, None], z.detach(), best_z)
        stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
        active_flag = active_flag & (stale < patience)
        if not active_flag.any():
            break

        loss = (lp * active_flag.float()).sum() / active_flag.float().sum().clamp_min(1.0)
        loss.backward()
        opt.step()

    return best_z.detach(), best_loss.detach(), steps_run


def _restart_inits(
    n: int,
    active_idx: List[int],
    prev_v: Optional[np.ndarray],
    R: int,
    opt_seed: int,
    round_id: int,
    support_code: int,
) -> np.ndarray:
    """Return restart initializations ``z`` for the whole split: [N, R, a].

    Restart order (guide 7.3): 0 -> v=0; 1 -> previous-round selected v projected
    to this support (seeded random in round 0); 2..R-1 -> seeded uniform random
    starts.  Random starts are drawn per (opt_seed, round, support, restart) from a
    single RNG stream, so the i-th sample's draw is deterministic and independent
    of chunking.
    """
    a = len(active_idx)
    z = np.zeros((n, R, a), dtype=np.float32)

    def uniform(restart_idx: int) -> np.ndarray:
        rng = np.random.default_rng(mix_seed(opt_seed, round_id, support_code, restart_idx))
        return rng.uniform(-1.5, 1.5, size=(n, a)).astype(np.float32)

    if prev_v is not None:
        v_proj = prev_v[:, active_idx].astype(np.float32)
        v_proj = np.clip(v_proj / 1.5, -0.999999, 0.999999)
        z[:, 1, :] = np.arctanh(v_proj).astype(np.float32)
    else:
        z[:, 1, :] = uniform(1)
    for r in range(2, R):
        z[:, r, :] = uniform(r)
    return z


def exact_support_e_step(
    bank: torch.nn.Module,
    S: np.ndarray,
    delta: np.ndarray,
    cfg: Mapping[str, Any],
    prev_v: Optional[np.ndarray] = None,
    round_id: int = 0,
    opt_seed: int = 0,
    audit_n: int = 1000,
    log: bool = False,
    return_all_supports: bool = False,
) -> Dict[str, np.ndarray]:
    """Run the full exact E-step over one split.

    Returns a dict with keys ``m`` [N,5] uint8, ``v`` [N,5] float32,
    ``effect_mse`` [N], ``penalized_J`` [N], ``second_margin`` [N],
    ``support_code`` [N] uint8, and ``audit_scores`` [audit_n, 32].
    """
    N = S.shape[0]
    R = cfg["restarts"]
    chunk = cfg.get("chunk_size", 4096)
    tie_eps = cfg["tie_epsilon"]

    S_dev = torch.from_numpy(S).to(DEVICE)
    d_dev = torch.from_numpy(delta).to(DEVICE)
    # Sample-major tiling across restarts: S_t[c*R + r] == S[c].
    S_t = S_dev.repeat_interleave(R, dim=0)
    d_t = d_dev.repeat_interleave(R, dim=0)

    scores = np.empty((N, NUM_SUPPORTS), dtype=np.float32)
    v_all = np.zeros((N, NUM_SUPPORTS, 5), dtype=np.float32)

    # Empty support (code 0): prediction == 0, v == 0, no inner solve.
    empty_mse = (delta ** 2).mean(axis=(-2, -1)).astype(np.float32)
    scores[:, 0] = empty_mse  # + 0 * LAMBDA_P

    started = time.time()
    inner_steps = 0
    for code in range(1, NUM_SUPPORTS):
        support = SUPPORTS[code]
        active_idx = [int(j) for j in np.flatnonzero(support)]
        a = len(active_idx)
        inits = _restart_inits(N, active_idx, prev_v, R, opt_seed, round_id, code)

        for c_start in range(0, N, chunk):
            c_end = min(c_start + chunk, N)
            C = c_end - c_start
            sl = slice(c_start * R, c_end * R)
            z_inits = torch.from_numpy(inits[c_start:c_end].transpose(1, 0, 2).reshape(C * R, a)).to(DEVICE)
            best_z, best_loss, steps_run = _adam_active_v(bank, S_t[sl], d_t[sl], active_idx, z_inits, cfg)
            inner_steps += steps_run * (C * R)
            best_z = best_z.reshape(C, R, a)
            best_loss = best_loss.reshape(C, R)
            best_r = best_loss.argmin(dim=1)  # [C]
            v_best = 1.5 * torch.tanh(best_z[torch.arange(C), best_r])  # [C,a]
            effect_mse = best_loss[torch.arange(C), best_r].cpu().numpy()
            scores[c_start:c_end, code] = effect_mse + LAMBDA_P * SUPPORT_SIZES[code]
            v_all[c_start:c_end, code, active_idx] = v_best.cpu().numpy()

        if log:
            print(
                f"  [E-step] round={round_id} support={code:02d}/{NUM_SUPPORTS-1} "
                f"(|s|={a}) elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    # Per-sample selection with tie-break; second-best margin from sorted scores.
    codes = np.arange(NUM_SUPPORTS, dtype=np.int64)
    best_code = tie_break_argmin(scores, SUPPORT_SIZES.astype(np.float64), codes.astype(np.float64), tie_eps)
    best_J = scores[np.arange(N), best_code]
    sorted_scores = np.sort(scores, axis=1)
    second_J = sorted_scores[:, 1]
    margin = (second_J - best_J).astype(np.float32)

    m = SUPPORTS[best_code].astype(np.uint8)
    v = v_all[np.arange(N), best_code, :].astype(np.float32)
    effect_mse = (best_J - LAMBDA_P * SUPPORT_SIZES[best_code]).astype(np.float32)

    audit_scores = scores[:audit_n].astype(np.float32)

    result = {
        "m": m,
        "v": v,
        "effect_mse": effect_mse,
        "penalized_J": best_J.astype(np.float32),
        "second_margin": margin,
        "support_code": best_code.astype(np.uint8),
        "audit_scores": audit_scores,
        "inner_steps": inner_steps,
    }
    if return_all_supports:
        result["all_support_scores"] = scores.astype(np.float32)
        result["all_support_v"] = v_all.astype(np.float32)
    return result
