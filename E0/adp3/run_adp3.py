"""E0-ADP3: learn mechanism identity from controlled contrast fragments.

The learner-facing loader returns only (group_id, S, d). Ground-truth switch
identity, realization and effect are stored separately and are opened only by
the evaluator. A1 uses hard single-candidate assignment and alternating fits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_e0 as base

CONFIG_PATH = ROOT / "configs" / "e0_adp3_8gb.json"
OUT = ROOT / "outputs" / "e0_adp3"
DATA = OUT / "contrast_data"
DEVICE = base.DEVICE


def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    raise TypeError(type(x))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def metadata(c: dict[str, Any], stage: str) -> dict[str, Any]:
    meta = base.environment_metadata()
    meta.update({
        "stage": stage,
        "world_seed": base.WORLD_SEED,
        "dataset_seed": base.DATASET_SEED,
        "contrast_seed": c["contrast_seed"],
        "optimization_seed": c["optimization_seed"],
        "config": c,
        "config_sha256": base.sha256_file(CONFIG_PATH),
        "script_sha256": base.sha256_file(Path(__file__)),
        "git_commit": base.git_commit(),
    })
    return meta


def all_positive_edges() -> list[tuple[int, int, int]]:
    """Return (low mask, high mask, switched GT index), always 0 -> 1."""
    edges: list[tuple[int, int, int]] = []
    for j in range(3):
        bit = 1 << (2 - j)
        for low in range(8):
            if low & bit == 0:
                edges.append((low, low | bit, j))
    assert len(edges) == 12
    return edges


def bits(code: int) -> np.ndarray:
    return np.asarray([(code >> 2) & 1, (code >> 1) & 1, code & 1], dtype=np.float32)


def generate_one_split(split: str, n_base: int, c: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    split_id = {"train": 0, "val": 1, "test": 2}[split]
    rng = np.random.default_rng(c["contrast_seed"] + 10007 * split_id)
    # Reuse the serialized original E0-A world. Besides enforcing exact world
    # identity, this avoids rebuilding it through a platform BLAS/SVD path.
    world = base.load_world("e0a")
    S_base = rng.uniform(-1.0, 1.0, size=(n_base, 3, 2)).astype(np.float32)
    m_on = np.ones((n_base, 3), dtype=np.float32)
    v_base = base.sample_realizations(rng, m_on)
    edges = all_positive_edges()
    per_base = int(c["contrasts_per_base"])
    rows: dict[str, list[np.ndarray | int]] = {k: [] for k in ("S", "d", "group_id", "fragment_id")}
    hidden: dict[str, list[np.ndarray | int | float]] = {k: [] for k in ("y_gt", "v_gt", "e_gt", "low_mask", "high_mask")}
    max_mse = 0.0
    fragment_id = 0
    for g in range(n_base):
        Sg = S_base[g:g+1]
        vg = v_base[g:g+1]
        deltas: list[np.ndarray] = []
        effects: list[np.ndarray] = []
        for code in range(8):
            mg = bits(code)[None]
            eg = base.compute_effects(Sg, mg, vg, world)
            effects.append(eg[0])
            deltas.append(eg.sum(axis=1)[0])
        chosen = rng.choice(len(edges), size=per_base, replace=False)
        for edge_idx in chosen:
            low, high, j = edges[int(edge_idx)]
            d = (deltas[high] - deltas[low]).astype(np.float32)
            oracle = effects[high][j]
            mse = float(np.mean((d - oracle) ** 2))
            max_mse = max(max_mse, mse)
            rows["S"].append(Sg[0]); rows["d"].append(d)
            rows["group_id"].append(g); rows["fragment_id"].append(fragment_id)
            hidden["y_gt"].append(j); hidden["v_gt"].append(float(vg[0, j])); hidden["e_gt"].append(oracle)
            hidden["low_mask"].append(low); hidden["high_mask"].append(high)
            fragment_id += 1
    visible = {
        "S": np.asarray(rows["S"], dtype=np.float32),
        "d": np.asarray(rows["d"], dtype=np.float32),
        "group_id": np.asarray(rows["group_id"], dtype=np.int32),
        "fragment_id": np.asarray(rows["fragment_id"], dtype=np.int32),
    }
    evaluator = {
        "y_gt": np.asarray(hidden["y_gt"], dtype=np.int8),
        "v_gt": np.asarray(hidden["v_gt"], dtype=np.float32),
        "e_gt": np.asarray(hidden["e_gt"], dtype=np.float32),
        "low_mask": np.asarray(hidden["low_mask"], dtype=np.int8),
        "high_mask": np.asarray(hidden["high_mask"], dtype=np.int8),
    }
    return visible, evaluator, max_mse


def stage_adp3_0(force: bool = False) -> dict[str, Any]:
    c = config()
    target = OUT / "adp3_0" / "final_metrics.json"
    if target.exists() and not force:
        return json.loads(target.read_text(encoding="utf-8"))
    DATA.mkdir(parents=True, exist_ok=True)
    hidden_all: dict[str, np.ndarray] = {}
    max_mse = 0.0
    counts: dict[str, int] = {}
    for split, n_base in c["base_groups"].items():
        visible, hidden, split_max = generate_one_split(split, int(n_base), c)
        np.savez_compressed(DATA / f"visible_{split}.npz", **visible)
        for key, value in hidden.items():
            hidden_all[f"{split}_{key}"] = value
        max_mse = max(max_mse, split_max)
        counts[split] = len(visible["S"])
    np.savez_compressed(DATA / "hidden_gt.npz", **hidden_all)
    result = {
        **metadata(c, "adp3_0"),
        "learner_visible_fields": ["group_id", "fragment_id", "S", "d"],
        "evaluator_only_file": "hidden_gt.npz",
        "fragment_counts": counts,
        "max_fragment_mse": max_mse,
        "pass_threshold": 1e-10,
        "pass": bool(max_mse < 1e-10),
    }
    dump_json(target, result)
    dump_json(OUT / "configs" / "resolved_config.json", c)
    print(json.dumps({"stage": "adp3_0", "max_fragment_mse": max_mse, "counts": counts, "pass": result["pass"]}, indent=2), flush=True)
    return result


def load_visible(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learner loader: deliberately cannot return GT identity or realization."""
    with np.load(DATA / f"visible_{split}.npz") as z:
        return z["S"], z["d"], z["group_id"]


def load_hidden(split: str) -> dict[str, np.ndarray]:
    """Evaluator-only loader."""
    with np.load(DATA / "hidden_gt.npz") as z:
        prefix = split + "_"
        return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}


def set_bank_trainable(bank: torch.nn.Module, trainable: bool) -> None:
    for p in bank.parameters():
        p.requires_grad_(trainable)


def fragment_e_step(bank: torch.nn.Module, S: np.ndarray, d: np.ndarray, c: dict[str, Any], seed: int) -> dict[str, np.ndarray | float]:
    """Optimize one scalar v for every fragment/candidate/restart, then hard-assign."""
    a = c["a1"]
    n, k, restarts = len(S), c["m_max"], a["inner_restarts"]
    all_err = np.empty((n, k), dtype=np.float32)
    all_v = np.empty((n, k), dtype=np.float32)
    restart_winner = np.empty((restarts, n), dtype=np.int16)
    set_bank_trainable(bank, False)
    bank.eval()
    started = time.time()
    for start in range(0, n, a["e_step_chunk_size"]):
        stop = min(n, start + a["e_step_chunk_size"])
        b = stop - start
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(int(seed + start * 1009))
        z = (0.35 * torch.randn((restarts, b, k), generator=generator, device=DEVICE)).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=a["inner_learning_rate"])
        St = torch.from_numpy(S[start:stop]).to(DEVICE)
        dt = torch.from_numpy(d[start:stop]).to(DEVICE)
        Sr = St.repeat(restarts, 1, 1)
        dr = dt.repeat(restarts, 1, 1)
        for _ in range(a["inner_steps"]):
            opt.zero_grad(set_to_none=True)
            v = a["v_bound"] * torch.tanh(z.reshape(restarts * b, k))
            pred = bank.raw_effects(Sr, v)
            error = (pred - dr[:, None]).square().mean(dim=(2, 3))
            error.sum().backward()
            opt.step()
        with torch.no_grad():
            v = a["v_bound"] * torch.tanh(z.reshape(restarts * b, k))
            pred = bank.raw_effects(Sr, v)
            error = (pred - dr[:, None]).square().mean(dim=(2, 3)).reshape(restarts, b, k)
            vv = v.reshape(restarts, b, k)
            chosen_restart = error.argmin(dim=0)
            cols_b = torch.arange(b, device=DEVICE)[:, None]
            cols_k = torch.arange(k, device=DEVICE)[None, :]
            best_err = error[chosen_restart, cols_b, cols_k]
            best_v = vv[chosen_restart, cols_b, cols_k]
            all_err[start:stop] = best_err.cpu().numpy()
            all_v[start:stop] = best_v.cpu().numpy()
            for rr in range(restarts):
                restart_winner[rr, start:stop] = error[rr].argmin(dim=1).cpu().numpy()
    assignment = all_err.argmin(axis=1).astype(np.int16)
    chosen_v = all_v[np.arange(n), assignment]
    agreement = float(np.mean(restart_winner[0] == restart_winner[1])) if restarts == 2 else float("nan")
    set_bank_trainable(bank, True)
    return {
        "assignment": assignment,
        "v": chosen_v.astype(np.float32),
        "v_all": all_v,
        "errors": all_err,
        "chosen_error": all_err[np.arange(n), assignment],
        "restart_assignment_agreement": agreement,
        "seconds": time.time() - started,
    }


def fragment_m_step(bank: torch.nn.Module, S: np.ndarray, d: np.ndarray, assignment: np.ndarray, v: np.ndarray,
                    c: dict[str, Any], optimizer: torch.optim.Optimizer, round_id: int) -> dict[str, float]:
    a = c["a1"]
    bank.train(); set_bank_trainable(bank, True)
    rng = np.random.default_rng(c["optimization_seed"] + 50021 * (round_id + 1))
    started = time.time(); losses: list[float] = []
    for _ in range(a["m_step_epochs_per_round"]):
        order = rng.permutation(len(S))
        for start in range(0, len(S), a["m_step_batch_size"]):
            idx = order[start:start + a["m_step_batch_size"]]
            St = torch.from_numpy(S[idx]).to(DEVICE)
            dt = torch.from_numpy(d[idx]).to(DEVICE)
            ct = torch.from_numpy(assignment[idx].astype(np.int64)).to(DEVICE)
            vt = torch.from_numpy(v[idx]).to(DEVICE)
            vall = torch.zeros((len(idx), c["m_max"]), device=DEVICE)
            vall.scatter_(1, ct[:, None], vt[:, None])
            optimizer.zero_grad(set_to_none=True)
            raw = bank.raw_effects(St, vall)
            pred = raw[torch.arange(len(idx), device=DEVICE), ct]
            loss = (pred - dt).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bank.parameters(), a["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {"seconds": time.time() - started, "mean_minibatch_mse": float(np.mean(losses))}


@torch.no_grad()
def assigned_prediction(bank: torch.nn.Module, S: np.ndarray, assignment: np.ndarray, v: np.ndarray, batch: int = 512) -> np.ndarray:
    out = []
    bank.eval()
    for start in range(0, len(S), batch):
        stop = min(len(S), start + batch)
        St = torch.from_numpy(S[start:stop]).to(DEVICE)
        ct = torch.from_numpy(assignment[start:stop].astype(np.int64)).to(DEVICE)
        vt = torch.from_numpy(v[start:stop]).to(DEVICE)
        vall = torch.zeros((stop-start, bank.m_max), device=DEVICE)
        vall.scatter_(1, ct[:, None], vt[:, None])
        raw = bank.raw_effects(St, vall)
        out.append(raw[torch.arange(stop-start, device=DEVICE), ct].cpu().numpy())
    return np.concatenate(out)


def r2(target: np.ndarray, pred: np.ndarray) -> float:
    ssr = float(np.sum((target - pred) ** 2))
    centered = target - target.mean(axis=0, keepdims=True)
    return 1.0 - ssr / max(float(np.sum(centered ** 2)), 1e-12)


@torch.no_grad()
def functional_matrix(bank: torch.nn.Module, v_all: np.ndarray, hidden: dict[str, np.ndarray], c: dict[str, Any]) -> np.ndarray:
    a, k = c["a1"], c["m_max"]
    rng = np.random.default_rng(7_777_003)
    n_probe = a["functional_probe_count"]
    S_probe = rng.uniform(-1.0, 1.0, size=(n_probe, 3, 2)).astype(np.float32)
    m_probe = np.ones((n_probe, 3), dtype=np.float32)
    v_probe = base.sample_realizations(rng, m_probe)
    world = base.load_world("e0a")
    mat = np.full((k, 3), -1e6, dtype=np.float64)
    for learned in range(k):
        for gt in range(3):
            take = hidden["y_gt"] == gt
            x, y = hidden["v_gt"][take], v_all[take, learned]
            # Closed-form scalar affine calibration avoids a platform LAPACK
            # dependency and is exactly equivalent to least squares here.
            x_mean, y_mean = float(x.mean()), float(y.mean())
            slope = float(np.sum((x - x_mean) * (y - y_mean)) / max(float(np.sum((x - x_mean) ** 2)), 1e-12))
            intercept = y_mean - slope * x_mean
            learned_parts = []
            for start in range(0, n_probe, 512):
                stop = min(n_probe, start + 512)
                vv = np.zeros((stop-start, k), dtype=np.float32)
                vv[:, learned] = slope * v_probe[start:stop, gt] + intercept
                raw = bank.raw_effects(torch.from_numpy(S_probe[start:stop]).to(DEVICE), torch.from_numpy(vv).to(DEVICE))
                learned_parts.append(raw[:, learned].cpu().numpy())
            pred = np.concatenate(learned_parts)
            mg = np.zeros((n_probe, 3), dtype=np.float32); mg[:, gt] = 1
            vg = np.zeros((n_probe, 3), dtype=np.float32); vg[:, gt] = v_probe[:, gt]
            target = base.compute_effects(S_probe, mg, vg, world)[:, gt]
            mat[learned, gt] = r2(target, pred)
    return mat


def match_metrics(func: np.ndarray, assignment: np.ndarray, y_gt: np.ndarray) -> dict[str, Any]:
    learned, gt = linear_sum_assignment(-np.nan_to_num(func, nan=-1e6))
    functional_gt_to_learned = {int(g): int(l) for l, g in zip(learned, gt)}
    matched = [float(func[functional_gt_to_learned[g], g]) for g in range(3)]

    # Clustering accuracy must be matched from the assignment contingency,
    # independently of the evaluator-only functional probe matching.
    contingency = np.zeros((func.shape[0], 3), dtype=np.int64)
    for candidate, truth in zip(assignment, y_gt):
        contingency[int(candidate), int(truth)] += 1
    cluster_learned, cluster_gt = linear_sum_assignment(-contingency)
    cluster_candidate_to_gt = {int(l): int(g) for l, g in zip(cluster_learned, cluster_gt)}
    predicted_gt = np.asarray([cluster_candidate_to_gt.get(int(x), -1) for x in assignment], dtype=np.int16)
    confusion = np.zeros((3, 4), dtype=np.int64)
    for truth, pred in zip(y_gt, predicted_gt):
        confusion[int(truth), int(pred) if pred >= 0 else 3] += 1

    functional_predicted_gt = np.full(len(assignment), -1, dtype=np.int16)
    for g, l in functional_gt_to_learned.items():
        functional_predicted_gt[assignment == l] = g
    return {
        "functional_hungarian_gt_to_candidate": functional_gt_to_learned,
        "assignment_hungarian_candidate_to_gt": cluster_candidate_to_gt,
        "matched_functional_r2": matched,
        "matched_functional_r2_min": min(matched),
        "fragment_matching_accuracy": float(np.mean(predicted_gt == y_gt)),
        "functional_mapping_fragment_accuracy": float(np.mean(functional_predicted_gt == y_gt)),
        "assignment_contingency_rows_candidate_cols_gt": contingency.tolist(),
        "fragment_confusion_matrix_rows_gt_cols_pred012extra": confusion.tolist(),
    }


def stage_a1(force: bool = False, rounds_override: int | None = None) -> dict[str, Any]:
    c = config()
    pre = stage_adp3_0(False)
    if not pre["pass"]:
        raise RuntimeError("STOP: Adp3-0 failed")
    out = OUT / ("a1_probe_seed_0" if rounds_override is not None else "a1_seed_0")
    final_path = out / "final_metrics.json"
    if final_path.exists() and not force and rounds_override is None:
        return json.loads(final_path.read_text(encoding="utf-8"))
    if force:
        # Preserve old evidence by refusing to delete; caller must use a clean output or explicitly archive it.
        if (out / "round_metrics.jsonl").exists():
            raise FileExistsError(f"Refusing destructive overwrite of {out}; archive it first")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "fragment_assignments").mkdir(parents=True, exist_ok=True)
    log = out / "round_metrics.jsonl"
    S_train, d_train, _ = load_visible("train")
    S_val, d_val, _ = load_visible("val")
    h_val = load_hidden("val")
    base.seed_everything(c["optimization_seed"])
    bank = base.SetMechanismBank(c["m_max"], c["mechanism_dim"], c["mechanism_heads"]).to(DEVICE)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=c["a1"]["m_step_learning_rate"],
                                  weight_decay=c["a1"]["m_step_weight_decay"])
    rounds = rounds_override if rounds_override is not None else c["a1"]["rounds"]
    previous: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    run_started = time.time()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for round_id in range(rounds):
        train_lat = fragment_e_step(bank, S_train, d_train, c, c["optimization_seed"] + round_id * 1000003)
        assignment = train_lat["assignment"]
        change = 1.0 if previous is None else float(np.mean(assignment != previous))
        mstat = fragment_m_step(bank, S_train, d_train, assignment, train_lat["v"], c, optimizer, round_id)
        train_pred = assigned_prediction(bank, S_train, assignment, train_lat["v"])
        train_mse = float(np.mean((train_pred - d_train) ** 2))
        val_lat = fragment_e_step(bank, S_val, d_val, c, c["optimization_seed"] + 70000001 + round_id * 1000003)
        val_pred = assigned_prediction(bank, S_val, val_lat["assignment"], val_lat["v"])
        val_nrmse = base.nrmse(d_val, val_pred)
        func = functional_matrix(bank, val_lat["v_all"], h_val, c)
        matched = match_metrics(func, val_lat["assignment"], h_val["y_gt"])
        counts = np.bincount(assignment, minlength=c["m_max"])
        val_counts = np.bincount(val_lat["assignment"], minlength=c["m_max"])
        heldout = [float(val_lat["errors"][val_lat["assignment"] == j, j].mean()) if val_counts[j] else None for j in range(c["m_max"])]
        effect_norm = []
        for j in range(c["m_max"]):
            take = val_lat["assignment"] == j
            effect_norm.append(float(np.linalg.norm(val_pred[take], axis=(1, 2)).mean()) if take.any() else None)
        torch.save(bank.state_dict(), out / "checkpoints" / f"round_{round_id:03d}_bank.pt")
        np.savez_compressed(out / "fragment_assignments" / f"round_{round_id:03d}.npz",
                            candidate=assignment, v=train_lat["v"], chosen_error=train_lat["chosen_error"])
        row = {
            "round": round_id,
            "train_fragment_mse": train_mse,
            "val_fragment_nrmse": val_nrmse,
            "candidate_usage": (counts / len(assignment)).tolist(),
            "candidate_sample_count": counts.tolist(),
            "candidate_heldout_error": heldout,
            "candidate_effect_norm": effect_norm,
            "assignment_change_rate": change,
            "exact_assignment_agreement": train_lat["restart_assignment_agreement"],
            "functional_r2_matrix": func.tolist(),
            **matched,
            "alive": [True] * c["m_max"],
            "num_alive": c["m_max"],
            "merge_proposals": [], "merge_accepts": [], "merge_seconds": 0.0,
            "e_step_seconds": train_lat["seconds"] + val_lat["seconds"],
            "m_step_seconds": mstat["seconds"],
            "round_seconds": train_lat["seconds"] + val_lat["seconds"] + mstat["seconds"],
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else 0.0,
        }
        append_jsonl(log, row); rows.append(row)
        print(f"[A1] r={round_id:02d} nrmse={val_nrmse:.4f} acc={matched['fragment_matching_accuracy']:.4f} "
              f"func={np.round(matched['matched_functional_r2'],3).tolist()} change={change:.3f} "
              f"usage={np.round(row['candidate_usage'],3).tolist()} e={row['e_step_seconds']:.1f}s m={row['m_step_seconds']:.1f}s", flush=True)
        previous = assignment.copy()
    last = rows[-1]
    last3_change = float(np.mean([x["assignment_change_rate"] for x in rows[-3:]]))
    th = c["pass_thresholds"]
    passed = bool(last["fragment_matching_accuracy"] > th["fragment_matching_accuracy_strict_gt"]
                  and all(x > th["matched_functional_r2_each"] for x in last["matched_functional_r2"])
                  and min(last["matched_functional_r2"]) > th["matched_functional_r2_min"]
                  and last3_change < th["last3_assignment_change_rate"])
    result = {
        **metadata(c, "adp3_a1"),
        "rounds_completed": rounds,
        "development_probe_only": rounds_override is not None,
        "last_round": last,
        "last3_assignment_change_rate": last3_change,
        "pass": passed,
        "stop_required": not passed,
        "wall_seconds": time.time() - run_started,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else 0.0,
    }
    # A short probe must not masquerade as final A1.
    destination = out / ("probe_metrics.json" if rounds_override is not None else "final_metrics.json")
    dump_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["adp3_0", "adp3_a1"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--probe-rounds", type=int)
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("Development implementation currently fixes optimization seed 0")
    result = stage_adp3_0(args.force) if args.stage == "adp3_0" else stage_a1(args.force, args.probe_rounds)
    print(json.dumps({k: result[k] for k in ("stage", "pass") if k in result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
