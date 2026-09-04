"""Read-only correction/audit of A1 evaluator metrics from saved checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import torch

E0_ROOT = Path(__file__).resolve().parent.parent
if str(E0_ROOT) not in sys.path:
    sys.path.insert(0, str(E0_ROOT))
from adp3.run_adp3 import OUT, config, dump_json, fragment_e_step, functional_matrix, load_hidden, load_visible, match_metrics
import run_e0 as base


def main() -> None:
    c = config()
    source = OUT / "a1_seed_0"
    target = OUT / "a1_seed_0" / "corrected_evaluator_metrics.json"
    S, d, _ = load_visible("val")
    hidden = load_hidden("val")
    rows = []
    for round_id in range(c["a1"]["rounds"]):
        bank = base.SetMechanismBank(c["m_max"], c["mechanism_dim"], c["mechanism_heads"]).to(base.DEVICE)
        bank.load_state_dict(torch.load(source / "checkpoints" / f"round_{round_id:03d}_bank.pt",
                                        map_location=base.DEVICE, weights_only=True))
        lat = fragment_e_step(bank, S, d, c, c["optimization_seed"] + 70000001 + round_id * 1000003)
        func = functional_matrix(bank, lat["v_all"], hidden, c)
        metric = match_metrics(func, lat["assignment"], hidden["y_gt"])
        row = {"round": round_id, **metric}
        rows.append(row)
        print(f"[audit] r={round_id:02d} cluster_acc={metric['fragment_matching_accuracy']:.4f} "
              f"functional_map_acc={metric['functional_mapping_fragment_accuracy']:.4f} "
              f"func={np.round(metric['matched_functional_r2'],3).tolist()}", flush=True)
    th = c["pass_thresholds"]
    last = rows[-1]
    verdict = bool(last["fragment_matching_accuracy"] > th["fragment_matching_accuracy_strict_gt"]
                   and all(x > th["matched_functional_r2_each"] for x in last["matched_functional_r2"])
                   and min(last["matched_functional_r2"]) > th["matched_functional_r2_min"])
    dump_json(target, {"read_only_no_retraining": True, "rounds": rows, "final_pass_excluding_stability": verdict})
    print(json.dumps({"final": last, "pass_excluding_stability": verdict}, indent=2), flush=True)


if __name__ == "__main__":
    main()
