"""Diagnostic: does a STRONG M-step make discovery converge to the GT basin?

Runs the same freeze-E / freeze-M loop as ``_discover_seed`` but with a much
larger M-step budget (``steps_per_candidate_per_round = 200`` instead of 16), for
15 rounds from a fresh random init (seed 0).  If SCR collapses and functional-R2
-> 1, the 16-step budget is the bottleneck (scale, not bug).  If SCR stays ~0.2
and functional-R2 stays < 0, the procedure is stuck (bug / non-identifiability).
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
from adp1.evaluate_adp1 import _hungarian, _bank_predictions


class _BankOnly:
    def __init__(self, bank):
        self.bank = bank


def main() -> None:
    cfg = json.loads(Path("configs/e0_adp1_mini.json").read_text(encoding="utf-8"))
    cfg["m_step"]["steps_per_candidate_per_round"] = 200  # diagnostic: strong M-step
    subset_idx = _subset_indices(cfg)
    seed = 0
    e_cfg = _effective_e_step(cfg)

    S_train, _, delta_train = _load_subset("train", subset_idx)
    S_val, _, delta_val = _load_subset("val", subset_idx)

    base.seed_everything(seed)
    full_model = base.E0AModel(cfg["m_max"], 64, 2, 4).to(DEVICE)
    bank = full_model.bank
    optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                  weight_decay=cfg["m_step"]["weight_decay"])
    probe = make_probe()

    prev_m = None
    prev_v = None
    for round_id in range(15):
        freeze_bank(bank)
        train_lat = exact_support_e_step(bank, S_train, delta_train, e_cfg, prev_v=prev_v,
                                         round_id=round_id, opt_seed=seed)
        scr = float(np.mean(train_lat["m"] != prev_m)) if prev_m is not None else None
        J_val = None
        if round_id % 2 == 0:
            val_lat = exact_support_e_step(bank, S_val, delta_val, e_cfg, prev_v=None,
                                           round_id=round_id, opt_seed=seed)
            J_val = float(val_lat["penalized_J"].mean())

        unfreeze_bank(bank)
        shared_before = snapshot_shared(bank)
        m_step_round(bank, optimizer, train_lat["m"], train_lat["v"], S_train, delta_train,
                     cfg, round_id, seed, probe, shared_before)

        print(f"[diag-strong] round={round_id} J_train={float(train_lat['penalized_J'].mean()):.6f} "
              f"J_val={J_val} scr={scr} active={float(train_lat['m'].sum(axis=1).mean()):.3f}", flush=True)
        prev_m = train_lat["m"].copy()
        prev_v = train_lat["v"].copy()

    freeze_bank(bank)
    S_iid, _, delta_iid = _load_subset("test_iid", subset_idx)
    lat_iid = exact_support_e_step(bank, S_iid, delta_iid, e_cfg, prev_v=None, round_id=0, opt_seed=seed)
    hidden_iid = {k: v[subset_idx["test_iid"]] for k, v in base.load_hidden("e0a", "test_iid").items()}
    func, _, _ = base.functional_matrix(_BankOnly(bank), lat_iid["v"], hidden_iid)
    mapping = _hungarian(func)
    matched = [float(func[mapping[gt], gt]) for gt in range(3)]
    _, pred = _bank_predictions(bank, S_iid, lat_iid["m"], lat_iid["v"])
    nrmse = base.nrmse(delta_iid, pred)
    print(f"[diag-strong] FINAL iid_nrmse={nrmse:.4f} func_r2_matched={matched} mapping={mapping}", flush=True)


if __name__ == "__main__":
    main()
