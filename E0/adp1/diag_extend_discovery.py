"""Diagnostic: continue discovery from a trained bank to test bug-vs-scale.

Loads seed 0's ``best_bank.pt`` and runs 18 more EM rounds (rounds 12..29),
printing per-round SCR / train-MSE / val-MSE, then the functional-R2 alignment vs
GT on test_iid.  If SCR collapses and functional-R2 -> 1, the 12-round budget was
too small (scale); if SCR stays ~0.2 and functional-R2 stays < 0, the procedure is
stuck in a non-GT basin.
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

import run_e0 as base
from adp1._utils import DEVICE
from adp1.exact_e_step import exact_support_e_step, freeze_bank, unfreeze_bank
from adp1.residual_m_step import m_step_round, snapshot_shared, make_probe
from adp1.run_adp1_mini import _load_subset, _subset_indices, _effective_e_step


class _BankOnly:
    def __init__(self, bank):
        self.bank = bank


def main() -> None:
    cfg = json.loads(Path("configs/e0_adp1_mini.json").read_text(encoding="utf-8"))
    subset_idx = _subset_indices(cfg)
    seed = 0
    e_cfg = _effective_e_step(cfg)
    o_cfg = cfg["outer_loop"]

    S_train, _, delta_train = _load_subset("train", subset_idx)
    S_val, _, delta_val = _load_subset("val", subset_idx)

    bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
    bank.load_state_dict(torch.load("outputs/e0_adp1_mini/seed_0/adp1/best_bank.pt",
                                    map_location=DEVICE, weights_only=True))
    optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                  weight_decay=cfg["m_step"]["weight_decay"])
    probe = make_probe()

    prev_m = None
    prev_v = None
    start_round = 12
    n_extra = 18

    for round_id in range(start_round, start_round + n_extra):
        freeze_bank(bank)
        train_lat = exact_support_e_step(bank, S_train, delta_train, e_cfg, prev_v=prev_v,
                                         round_id=round_id, opt_seed=seed)
        scr = float(np.mean(train_lat["m"] != prev_m)) if prev_m is not None else None
        J_val = None
        if round_id % o_cfg["val_every_rounds"] == 0:
            val_lat = exact_support_e_step(bank, S_val, delta_val, e_cfg, prev_v=None,
                                           round_id=round_id, opt_seed=seed)
            J_val = float(val_lat["penalized_J"].mean())

        unfreeze_bank(bank)
        shared_before = snapshot_shared(bank)
        m_step_round(bank, optimizer, train_lat["m"], train_lat["v"], S_train, delta_train,
                     cfg, round_id, seed, probe, shared_before)

        print(f"[diag-extend] round={round_id} J_train={float(train_lat['penalized_J'].mean()):.6f} "
              f"J_val={J_val} scr={scr} active={float(train_lat['m'].sum(axis=1).mean()):.3f}", flush=True)
        prev_m = train_lat["m"].copy()
        prev_v = train_lat["v"].copy()

    # functional alignment on test_iid with the extended bank
    freeze_bank(bank)
    S_iid, _, delta_iid = _load_subset("test_iid", subset_idx)
    lat_iid = exact_support_e_step(bank, S_iid, delta_iid, e_cfg, prev_v=None, round_id=0, opt_seed=seed)
    hidden_iid = {k: v[subset_idx["test_iid"]] for k, v in base.load_hidden("e0a", "test_iid").items()}
    func, _, _ = base.functional_matrix(_BankOnly(bank), lat_iid["v"], hidden_iid)
    from adp1.evaluate_adp1 import _hungarian
    mapping = _hungarian(func)
    matched = [float(func[mapping[gt], gt]) for gt in range(3)]
    pred_nrmse = base.nrmse(delta_iid, None)  # placeholder; compute real below
    from adp1.evaluate_adp1 import _bank_predictions
    _, pred = _bank_predictions(bank, S_iid, lat_iid["m"], lat_iid["v"])
    pred_nrmse = base.nrmse(delta_iid, pred)
    print(f"[diag-extend] FINAL iid_nrmse={pred_nrmse:.4f} func_r2_matched={matched} mapping={mapping}", flush=True)


if __name__ == "__main__":
    main()
