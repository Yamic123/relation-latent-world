"""Diagnose WHY the mini solver-ladder gate fails (87.5% < 99% agreement).

For the 16 fixed calibration samples, print the top-2 supports' penalized J and
their gap under BOTH the Adam inner solver and the 10-start L-BFGS-B reference,
and test whether more solver effort (more Adam steps/restarts, more L-BFGS-B
starts) flips the discrete support.  This distinguishes (a) near-tie supports
(gap ~ fp32 noise) from (b) a genuine convergence failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

_E0 = Path(__file__).resolve().parent.parent
if str(_E0) not in sys.path:
    sys.path.insert(0, str(_E0))

import json
import numpy as np
import torch
from scipy.optimize import minimize

import run_e0 as base
from adp1._utils import DEVICE, LAMBDA_P, NUM_SUPPORTS, SUPPORTS, SUPPORT_SIZES, tie_break_argmin
from adp1.exact_e_step import exact_support_e_step, freeze_bank
from adp1.run_adp1_mini import _lbfgsb_reference

cfg = json.loads(Path("configs/e0_adp1_mini.json").read_text())
N = 16
base.seed_everything(0)
bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
freeze_bank(bank)
S, _, delta = base.load_visible("e0a", "train", "identity")
S, delta = S[:N], delta[:N]


def top2(scores: np.ndarray) -> np.ndarray:
    # scores [N,32] -> indices of two smallest (by tie-break then argpartition)
    out = np.empty((scores.shape[0], 2), dtype=np.int64)
    codes = np.arange(NUM_SUPPORTS, dtype=np.int64)
    for i in range(scores.shape[0]):
        order = np.lexsort((codes, SUPPORT_SIZES, scores[i]))
        out[i] = order[:2]
    return out


ref = _lbfgsb_reference(bank, S, delta, n_starts=10)
J_ref = np.full((N, NUM_SUPPORTS), np.nan)
# recompute per-sample full matrix from reference internals is expensive; instead
# recompute the two competing supports directly below.

for label, (steps, restarts) in {"C(30/3)": (30, 3), "X(100/8)": (100, 8)}.items():
    e_cfg = dict(cfg["e_step"]); e_cfg["max_steps"] = steps; e_cfg["restarts"] = restarts
    lat = exact_support_e_step(bank, S, delta, e_cfg, prev_v=None, round_id=0, opt_seed=0)
    adam_scores = lat["audit_scores"][:N].astype(np.float64)  # [N,32]
    adam_sup = lat["support_code"].astype(np.int64)[:N]

    # Full L-BFGS-B J matrix for a clean comparison (16x31 minimizes).
    J_all = np.full((N, NUM_SUPPORTS), np.inf)
    bank_cpu = type(bank)(bank.m_max, bank.dim, bank.heads)
    bank_cpu.load_state_dict({k: v.cpu() for k, v in bank.state_dict().items()})
    bank_cpu.eval()
    S_t = torch.from_numpy(S); d_t = torch.from_numpy(delta)
    for i in range(N):
        Si = S_t[i:i+1]; di = d_t[i:i+1]
        for code in range(NUM_SUPPORTS):
            support = SUPPORTS[code]; active_idx = [int(j) for j in np.flatnonzero(support)]; a = len(active_idx)
            mask = torch.zeros(5); mask[active_idx] = 1.0
            if a == 0:
                bo = float(((torch.zeros_like(di) - di) ** 2).mean())
            else:
                def objective(va):
                    v = torch.zeros(1, 5); v[:, active_idx] = torch.from_numpy(va.astype(np.float32))
                    with torch.no_grad():
                        raw = bank_cpu.raw_effects(Si, v)
                        pred = (raw * mask[None, :, None, None]).sum(dim=1)
                    return float(((pred - di) ** 2).mean())
                rng = np.random.default_rng(7_000_000 + i * NUM_SUPPORTS + code)
                starts = [np.zeros(a, np.float32)]
                starts.extend(rng.uniform(-1.5, 1.5, size=(9, a)).astype(np.float32))
                bo = np.inf
                for x0 in starts:
                    r = minimize(objective, x0, method="L-BFGS-B", bounds=[(-1.5, 1.5)] * a,
                                 options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8})
                    bo = min(bo, float(r.fun))
            J_all[i, code] = bo + LAMBDA_P * SUPPORT_SIZES[code]
    lbf_sup = tie_break_argmin(J_all, SUPPORT_SIZES.astype(np.float64), np.arange(NUM_SUPPORTS, dtype=np.float64), 1e-8)

    print(f"\n=== {label} ===")
    for i in range(N):
        t2 = top2(adam_scores[i:i+1])[0]
        a1, a2 = int(t2[0]), int(t2[1])
        gap_a = adam_scores[i, a2] - adam_scores[i, a1]
        l1, l2 = int(np.argsort(J_all[i])[0]), int(np.argsort(J_all[i])[1])
        # find L-BFGS-B best two by full sort
        order_l = np.lexsort((np.arange(NUM_SUPPORTS), SUPPORT_SIZES, J_all[i]))
        ll1, ll2 = int(order_l[0]), int(order_l[1])
        gap_l = J_all[i, ll2] - J_all[i, ll1]
        agree = adam_sup[i] == lbf_sup[i]
        print(f"  i={i:2d} adam={adam_sup[i]:02d} lbf={lbf_sup[i]:02d} agree={int(agree)} "
              f"| adam_top2=({a1:02d},{a2:02d}) gap={gap_a:.3e} | lbf_top2=({ll1:02d},{ll2:02d}) gap={gap_l:.3e}")
