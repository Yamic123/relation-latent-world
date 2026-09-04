"""Diagnostic: does the continuous-solver agreement recover once the bank is
trained (mechanisms distinct)?  Trains a bank for a few cheap M-step rounds on a
small subset, then re-runs the formal Adam-vs-L-BFGS-B sanity comparison."""

import json
import sys
from pathlib import Path

_E0 = Path(__file__).resolve().parent.parent
if str(_E0) not in sys.path:
    sys.path.insert(0, str(_E0))

import numpy as np
import torch
from scipy.optimize import minimize

import run_e0 as base
from adp1._utils import DEVICE, NUM_SUPPORTS, SUPPORTS, SUPPORT_SIZES, tie_break_argmin
from adp1.exact_e_step import exact_support_e_step, freeze_bank, unfreeze_bank
from adp1.residual_m_step import m_step_round, snapshot_shared, make_probe

torch.backends.cuda.matmul.allow_tf32 = False
cfg = json.loads(Path("configs/e0_adp1.json").read_text())

# --- train a bank cheaply on a small subset ---
base.seed_everything(0)
model = base.E0AModel(cfg["m_max"], 64, 2, 4)
bank = model.bank.to(DEVICE)
del model
S, _, delta = base.load_visible("e0a", "train", "identity")
Ssub, dsub = S[:400], delta[:400]
opt = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                        weight_decay=cfg["m_step"]["weight_decay"])
probe = make_probe()
fast_e = dict(cfg["e_step"]); fast_e["max_steps"] = 40; fast_e["restarts"] = 3
prev_v = None
for r in range(5):
    freeze_bank(bank)
    lat = exact_support_e_step(bank, Ssub, dsub, fast_e, prev_v=prev_v, round_id=r, opt_seed=0)
    unfreeze_bank(bank)
    sb = snapshot_shared(bank)
    m_step_round(bank, opt, lat["m"], lat["v"], Ssub, dsub, cfg, r, 0, probe, sb)
    prev_v = lat["v"]
    print(f"round {r}: effect_mse={lat['effect_mse'].mean():.4f} active={lat['m'].mean():.3f}", flush=True)

# --- formal sanity comparison on the trained bank (100 samples) ---
freeze_bank(bank)
N = 100
Sn, dn = S[:N], delta[:N]
Sn_t = torch.from_numpy(Sn).to(DEVICE)
dn_t = torch.from_numpy(dn).to(DEVICE)


def adam_compare():
    lat = exact_support_e_step(bank, Sn, dn, cfg["e_step"], round_id=0, opt_seed=0)
    return lat["support_code"], lat["penalized_J"].astype(np.float64)


def lbfgsb_compare():
    bank_cpu = type(bank)(bank.m_max, bank.dim, bank.heads)
    bank_cpu.load_state_dict({k: v.cpu() for k, v in bank.state_dict().items()})
    bank_cpu.eval()
    J_all = np.full((N, NUM_SUPPORTS), np.inf, dtype=np.float64)
    for i in range(N):
        Si = torch.from_numpy(Sn[i:i + 1]); di = torch.from_numpy(dn[i:i + 1])
        for code in range(NUM_SUPPORTS):
            support = SUPPORTS[code]
            active_idx = [int(j) for j in np.flatnonzero(support)]
            a = len(active_idx)
            mask = torch.zeros(5); mask[active_idx] = 1.0
            if a == 0:
                best_obj = float(((torch.zeros_like(di) - di) ** 2).mean())
            else:
                def objective(v_active):
                    v = torch.zeros(1, 5); v[:, active_idx] = torch.from_numpy(v_active.astype(np.float32))
                    with torch.no_grad():
                        raw = bank_cpu.raw_effects(Si, v)
                        pred = (raw * mask[None, :, None, None]).sum(dim=1)
                    return float(((pred - di) ** 2).mean())
                bounds = [(-1.5, 1.5)] * a
                rng = np.random.default_rng(7_000_000 + i * NUM_SUPPORTS + code)
                starts = [np.zeros(a, dtype=np.float32)]
                starts.extend(rng.uniform(-1.5, 1.5, size=(9, a)).astype(np.float32))
                best_obj = np.inf
                for x0 in starts:
                    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8})
                    best_obj = min(best_obj, float(res.fun))
            J_all[i, code] = best_obj + 1e-3 * SUPPORT_SIZES[code]
    codes = np.arange(NUM_SUPPORTS, dtype=np.int64)
    best_support = tie_break_argmin(J_all, SUPPORT_SIZES.astype(np.float64), codes.astype(np.float64), 1e-8)
    return best_support, J_all[np.arange(N), best_support]


a_sup, a_J = adam_compare()
l_sup, l_J = lbfgsb_compare()
agreement = float(np.mean(a_sup == l_sup))
median_dJ = float(np.median(np.abs(a_J - l_J)))
max_dJ = float(np.max(np.abs(a_J - l_J)))
print(f"\nTRAINED-BANK sanity: support_agreement={agreement:.3f} median|dJ|={median_dJ:.2e} max|dJ|={max_dJ:.2e}", flush=True)
