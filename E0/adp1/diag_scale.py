"""Diagnostic: is the non-convergence a SAMPLE-SIZE (identifiability) issue?

Runs the freeze-E / freeze-M discovery loop from random init with a larger train
subset (``--n-train``, default 4096) and a strong M-step (200 steps/candidate),
for 15 rounds, then reports SCR + functional-R2 alignment vs GT.  If SCR collapses
and functional-R2 -> 1 as N grows, the mini 1024-sample budget is the bottleneck.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_E0 = Path(__file__).resolve().parent.parent
if str(_E0) not in sys.path:
    sys.path.insert(0, str(_E0))

import json
import numpy as np
import torch

import run_e0 as base
from adp1._utils import DEVICE
from adp1.exact_e_step import exact_support_e_step, freeze_bank, unfreeze_bank
from adp1.residual_m_step import m_step_round, snapshot_shared, make_probe
from adp1.run_adp1_mini import _effective_e_step
from adp1.evaluate_adp1 import _hungarian, _bank_predictions


class _BankOnly:
    def __init__(self, bank):
        self.bank = bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--m-steps", type=int, default=200)
    args = ap.parse_args()

    cfg = json.loads(Path("configs/e0_adp1_mini.json").read_text(encoding="utf-8"))
    cfg["m_step"]["steps_per_candidate_per_round"] = args.m_steps
    seed = 0
    e_cfg = _effective_e_step(cfg)

    S, _, delta = base.load_visible("e0a", "train", "identity")
    S_train, delta_train = S[: args.n_train], delta[: args.n_train]

    base.seed_everything(seed)
    full_model = base.E0AModel(cfg["m_max"], 64, 2, 4).to(DEVICE)
    bank = full_model.bank
    optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                  weight_decay=cfg["m_step"]["weight_decay"])
    probe = make_probe()

    prev_m = None
    prev_v = None
    for round_id in range(args.rounds):
        freeze_bank(bank)
        train_lat = exact_support_e_step(bank, S_train, delta_train, e_cfg, prev_v=prev_v,
                                         round_id=round_id, opt_seed=seed)
        scr = float(np.mean(train_lat["m"] != prev_m)) if prev_m is not None else None

        unfreeze_bank(bank)
        shared_before = snapshot_shared(bank)
        m_step_round(bank, optimizer, train_lat["m"], train_lat["v"], S_train, delta_train,
                     cfg, round_id, seed, probe, shared_before)

        print(f"[diag-scale N={args.n_train}] round={round_id} "
              f"J_train={float(train_lat['penalized_J'].mean()):.6f} "
              f"scr={scr} active={float(train_lat['m'].sum(axis=1).mean()):.3f}", flush=True)
        prev_m = train_lat["m"].copy()
        prev_v = train_lat["v"].copy()

    freeze_bank(bank)
    S_iid, _, delta_iid = base.load_visible("e0a", "test_iid", "identity")
    S_iid, delta_iid = S_iid[:512], delta_iid[:512]
    lat_iid = exact_support_e_step(bank, S_iid, delta_iid, e_cfg, prev_v=None, round_id=0, opt_seed=seed)
    hidden_iid = base.load_hidden("e0a", "test_iid")
    hidden_iid = {k: v[:512] for k, v in hidden_iid.items()}
    func, _, _ = base.functional_matrix(_BankOnly(bank), lat_iid["v"], hidden_iid)
    mapping = _hungarian(func)
    matched = [float(func[mapping[gt], gt]) for gt in range(3)]
    _, pred = _bank_predictions(bank, S_iid, lat_iid["m"], lat_iid["v"])
    nrmse = base.nrmse(delta_iid, pred)
    print(f"[diag-scale N={args.n_train}] FINAL iid_nrmse={nrmse:.4f} "
          f"func_r2_matched={matched} mapping={mapping}", flush=True)


if __name__ == "__main__":
    main()
