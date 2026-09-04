"""Create preregistered ADP12 diagnostic figures from completed pilot outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The wm environment has a known Windows DLL failure in NumPy's native BLAS
# inversion path. Matplotlib only inverts tiny affine matrices here, so use a
# local scalar Gauss-Jordan implementation rather than changing the environment.
def _safe_inv(matrix):
    a = np.asarray(matrix, dtype=float)
    n = a.shape[0]
    aug = np.concatenate([a.copy(), np.eye(n)], axis=1)
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(aug[col:, col])))
        if pivot != col: aug[[col, pivot]] = aug[[pivot, col]]
        aug[col] /= aug[col, col]
        for row in range(n):
            if row != col: aug[row] -= aug[row, col] * aug[col]
    return aug[:, n:]

np.linalg.inv = _safe_inv
import matplotlib.transforms as _mtransforms
_mtransforms.inv = _safe_inv
def _safe_dot(left, right):
    x, y = np.asarray(left), np.asarray(right)
    if x.ndim == 2 and y.ndim == 2: return (x[:, :, None] * y[None, :, :]).sum(axis=1)
    if x.ndim == 2 and y.ndim == 1: return (x * y[None, :]).sum(axis=1)
    if x.ndim == 1 and y.ndim == 2: return (x[:, None] * y).sum(axis=0)
    if x.ndim == 1 and y.ndim == 1: return (x * y).sum()
    return np.tensordot(x, y, axes=([-1], [0]))
np.dot = _safe_dot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "e0_adp12_bayesian_responsibility"
FIG = OUT / "aggregate_3seed" / "figures"
VARIANTS = {
    "H1 soft / no death": "h1_soft_no_death",
    "H2 hard / evidence death": "h2_hard_evidence_death",
    "H3 soft / evidence death": "h3_soft_evidence_death",
}


def rows(folder: str, seed: int):
    path = OUT / folder / f"seed_{seed:03d}" / "training" / "round_metrics.jsonl"
    # Select the last record for a round so the analysis remains robust to an interrupted retry.
    by_round = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        by_round[x["round"]] = x
    return [by_round[k] for k in sorted(by_round)]


def weighted_mixing(row):
    count = np.asarray(row["candidate_family_count"], float)
    purity = np.asarray(row["candidate_purity_fixed"], float)
    good = np.isfinite(purity) & (count > 0)
    return 1.0 - float(np.sum(count[good] * purity[good]) / np.sum(count[good]))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors[:5]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    for col, (title, folder) in enumerate(VARIANTS.items()):
        for seed in range(3):
            data = rows(folder, seed)
            r = [x["round"] for x in data]
            rsp = np.asarray([x["ema_responsibility"] for x in data])
            ax = axes[seed, col]
            for j in range(5): ax.plot(r, rsp[:, j], color=colors[j], lw=1.5, label=f"C{j+1}")
            ax.set_title(f"{title}, seed {seed}")
            ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=5, fontsize=8)
    for ax in axes[-1]: ax.set_xlabel("round")
    for ax in axes[:, 0]: ax.set_ylabel("EMA responsibility")
    fig.tight_layout(); fig.savefig(FIG / "responsibility_trajectories.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    for col, (title, folder) in enumerate(VARIANTS.items()):
        for seed in range(3):
            data = rows(folder, seed); r = [x["round"] for x in data]
            ax = axes[seed, col]; ax.plot(r, [x["posterior_entropy_mean"] for x in data], color="#3366aa", label="entropy")
            ax2 = ax.twinx(); ax2.plot(r, [x["heldout_realization_nrmse"] for x in data], color="#cc3311", alpha=.65, label="NRMSE")
            ax.set_title(f"{title}, seed {seed}"); ax.grid(alpha=.2)
            if col != 2: ax2.set_yticklabels([])
            else: ax2.set_ylabel("heldout NRMSE")
    for ax in axes[-1]: ax.set_xlabel("round")
    for ax in axes[:, 0]: ax.set_ylabel("posterior entropy")
    fig.tight_layout(); fig.savefig(FIG / "entropy_and_nrmse.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    variant_colors = ["#4477aa", "#cc3311", "#aa4499"]
    for color, (title, folder) in zip(variant_colors, VARIANTS.items()):
        for seed in range(3):
            data = rows(folder, seed)
            ax.plot([x["posterior_entropy_mean"] for x in data], [weighted_mixing(x) for x in data], color=color, lw=1.2, alpha=.65, label=title if seed == 0 else None)
    ax.set_xlabel("mean posterior entropy"); ax.set_ylabel("weighted family mixing (1 - purity)"); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "mixing_vs_posterior_uncertainty.png", dpi=180); plt.close(fig)

    print(json.dumps({"pass": True, "figure_dir": str(FIG), "files": sorted(p.name for p in FIG.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
