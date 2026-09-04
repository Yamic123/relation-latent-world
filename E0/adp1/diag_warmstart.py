"""Diagnostic: does the freeze-E / freeze-M procedure STAY in the GT basin when
initialized there?

This is the decisive warm-start test.  It reads ``hidden_gt.npz`` ONCE to *fit* a
bank whose candidate mechanisms 0/1/2 reproduce the GT mechanisms (and candidates
3/4 are pushed to zero output).  That fitted bank is then handed to the exact
same freeze-E / freeze-M loop used by discovery -- which sees ONLY ``S`` and
``delta_S`` (visible), never hidden GT.  If the recovered assignment collapses to
``m_gt``/``v_gt`` and functional-R2 / NRMSE stay ~1 through several rounds, then
(i) the implementation is correct and (ii) the GT basin is stable, so the
random-init failure is purely reachability / identifiability at the mini scale.
If the bank drifts away, there is a hidden bug.

Fit uses 8192 hidden train samples (a faithful GT mechanism); the freeze loop
under test uses the mini 1024-sample train subset, matching discovery exactly.
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
import torch.nn.functional as F

import run_e0 as base
from adp1._utils import DEVICE
from adp1.exact_e_step import exact_support_e_step, freeze_bank, unfreeze_bank
from adp1.residual_m_step import m_step_round, snapshot_shared, make_probe
from adp1.run_adp1_mini import _load_subset, _subset_indices, _effective_e_step
from adp1.evaluate_adp1 import _hungarian, _bank_predictions


class _BankOnly:
    def __init__(self, bank):
        self.bank = bank


def _fit_bank_to_gt(bank: torch.nn.Module, S: np.ndarray, v_gt: np.ndarray,
                    e_gt: np.ndarray, steps: int, lr: float) -> None:
    """Fit candidates 0..2 -> GT mechanisms, candidates 3..4 -> zero output."""
    N = S.shape[0]
    S_t = torch.from_numpy(S).to(DEVICE)
    v_in = torch.zeros(N, bank.m_max, device=DEVICE)
    v_in[:, :3] = torch.from_numpy(v_gt).to(DEVICE)          # v_gt: [N,3]
    target = torch.zeros(N, bank.m_max, 3, 2, device=DEVICE)
    target[:, :3] = torch.from_numpy(e_gt).to(DEVICE)         # e_gt: [N,3,3,2]

    opt = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=1e-5)
    for step in range(steps):
        opt.zero_grad()
        raw = bank.raw_effects(S_t, v_in)
        loss = F.mse_loss(raw, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                nrmse = base.nrmse(target.cpu().numpy(), bank.raw_effects(S_t, v_in).cpu().numpy())
            print(f"  [warmstart fit] step={step} loss={float(loss):.6e} nrmse={nrmse:.5f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-steps", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads(Path("configs/e0_adp1_mini.json").read_text(encoding="utf-8"))
    subset_idx = _subset_indices(cfg)
    e_cfg = _effective_e_step(cfg)

    # ---- Fit bank to GT (reads hidden GT; validate-style diagnostic) ----
    S_fit, _, _ = base.load_visible("e0a", "train", "identity")
    hidden_fit = base.load_hidden("e0a", "train")
    n_fit = 8192
    S_fit = S_fit[:n_fit]
    v_gt_fit = hidden_fit["v_gt"][:n_fit]
    e_gt_fit = hidden_fit["e_gt"][:n_fit]

    base.seed_everything(args.seed)
    bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
    _fit_bank_to_gt(bank, S_fit, v_gt_fit, e_gt_fit, args.fit_steps, lr=3e-3)

    # ---- Freeze loop under test (visible-only, mini 1024 subset) ----
    S_train, _, delta_train = _load_subset("train", subset_idx)
    hidden_train = base.load_hidden("e0a", "train")
    m_gt = hidden_train["m_gt"][subset_idx["train"]]
    v_gt = hidden_train["v_gt"][subset_idx["train"]]

    optimizer = torch.optim.AdamW(bank.parameters(), lr=cfg["m_step"]["learning_rate"],
                                  weight_decay=cfg["m_step"]["weight_decay"])
    probe = make_probe()

    for round_id in range(args.rounds):
        # --- E-step (bank frozen): does it recover m_gt / v_gt? ---
        freeze_bank(bank)
        lat = exact_support_e_step(bank, S_train, delta_train, e_cfg, prev_v=None,
                                   round_id=round_id, opt_seed=args.seed)
        m = lat["m"].astype(np.float32)          # [N,5]
        v = lat["v"]                              # [N,5]
        exact = float(np.mean(np.all(m[:, :3] == m_gt, axis=1)))
        f1s = [base.binary_metrics(m_gt[:, j], m[:, j])["f1"] for j in range(3)]
        v_r2s = []
        for j in range(3):
            act = m_gt[:, j] == 1
            v_r2s.append(base.r2_score(v_gt[act, j], v[act, j]))
        redundant_usage = float(max(m[:, 3].mean(), m[:, 4].mean()))

        # --- M-step (assignment frozen): does the bank stay at GT? ---
        unfreeze_bank(bank)
        shared_before = snapshot_shared(bank)
        m_step_round(bank, optimizer, lat["m"], lat["v"], S_train, delta_train,
                     cfg, round_id, args.seed, probe, shared_before)

        # post-M-step alignment on mini test_iid (hidden, validate only)
        freeze_bank(bank)
        S_iid, _, delta_iid = _load_subset("test_iid", subset_idx)
        lat_iid = exact_support_e_step(bank, S_iid, delta_iid, e_cfg, prev_v=None,
                                       round_id=0, opt_seed=args.seed)
        hidden_iid = {k: vv[subset_idx["test_iid"]] for k, vv in base.load_hidden("e0a", "test_iid").items()}
        func, _, _ = base.functional_matrix(_BankOnly(bank), lat_iid["v"], hidden_iid)
        mapping = _hungarian(func)
        func_matched = [float(func[mapping[gt], gt]) for gt in range(3)]
        _, pred = _bank_predictions(bank, S_iid, lat_iid["m"], lat_iid["v"])
        nrmse = base.nrmse(delta_iid, pred)

        print(f"[warmstart] round={round_id} exact_support={exact:.4f} "
              f"f1={[round(f, 3) for f in f1s]} v_r2={[round(r, 3) for r in v_r2s]} "
              f"redundant={redundant_usage:.3f} | postM iid_nrmse={nrmse:.4f} "
              f"func_r2={[round(r, 3) for r in func_matched]} mapping={mapping}", flush=True)


if __name__ == "__main__":
    main()
