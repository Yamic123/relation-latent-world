"""Re-run the continuous-solver sanity gate (Adam vs L-BFGS-B) against an
already-trained mechanism bank checkpoint.

The original ``validate-solver`` runs on a *randomly-initialized* bank, where the
continuous objective is multi-modal / has near-tie supports, so the support
agreement gate (>= 0.99) legitimately fails even though the objectives agree to
~1e-9.  This script re-runs the exact same comparison against a trained bank
(mechanisms specialized, GT mechanisms quadratic in v) to confirm the gate would
pass once the bank is trained -- see ``outputs/e0_adp1/solver_sanity/NOTE.md``.

Usage::

    python E0/adp1/recheck_solver_sanity.py --bank <path.pt> [--seed 0] [--num-samples 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_E0_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

if str(_E0_ROOT) not in sys.path:
    sys.path.insert(0, str(_E0_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from adp1 import run_adp1  # noqa: E402
from adp1._utils import ADP1_OUTPUT_ROOT, DEVICE, dump_json  # noqa: E402
from adp1.exact_e_step import exact_support_e_step, freeze_bank  # noqa: E402

import run_e0 as base  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="path to a trained bank .pt state_dict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = run_adp1.load_config()
    base.seed_everything(args.seed)
    bank = base.SetMechanismBank(cfg["m_max"], 64, 4).to(DEVICE)
    bank.load_state_dict(torch.load(args.bank, map_location=DEVICE, weights_only=True))
    freeze_bank(bank)

    S, _, delta = base.load_visible("e0a", "train", "identity")
    S = S[: args.num_samples]
    delta = delta[: args.num_samples]

    e_cfg = cfg["e_step"]
    latent = exact_support_e_step(bank, S, delta, e_cfg, prev_v=None, round_id=0, opt_seed=args.seed)
    adam_support = latent["support_code"].astype(np.int64)
    adam_J = latent["penalized_J"].astype(np.float64)

    ref = run_adp1._lbfgsb_reference(bank, S, delta)
    lbfgsb_support = ref["support"]
    lbfgsb_J = ref["J"]

    agreement = float(np.mean(adam_support == lbfgsb_support))
    j_diff = np.abs(adam_J - lbfgsb_J)
    median_diff = float(np.median(j_diff))

    result = {
        "bank": str(args.bank),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "support_agreement": agreement,
        "support_agreement_pass": bool(agreement >= 0.99),
        "median_abs_J_diff": median_diff,
        "median_J_diff_pass": bool(median_diff <= 1e-5),
        "max_abs_J_diff": float(np.max(j_diff)),
        "valid": bool(agreement >= 0.99 and median_diff <= 1e-5),
    }

    out_dir = Path(args.out_dir) if args.out_dir else (ADP1_OUTPUT_ROOT / "solver_sanity" / "trained_recheck")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
