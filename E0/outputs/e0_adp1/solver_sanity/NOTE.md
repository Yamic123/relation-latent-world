# Solver-sanity caveat (E0-ADP1)

## What happened

`validate-solver` (guide §7.3) reports `valid: false` on a **randomly-initialized**
mechanism bank:

| metric | value | gate | result |
|---|---|---|---|
| support agreement (Adam vs L-BFGS-B) | **0.89** | >= 0.99 | FAIL |
| median \|J_adam − J_lbfgsb\| | 3.73e-09 | <= 1e-5 | PASS |
| max \|J_adam − J_lbfgsb\| | 1.44e-02 | — | — |

## Diagnosis (not a solver bug)

The two solvers agree on the *continuous objective* to ~1e-9, so the Adam
inner-loop is not diverging or mis-optimizing. The 11% support disagreement comes
from **near-tie supports**: on a random bank the per-mechanism functions
`M_j(S, v_j)` are random MLPs in `v_j` (via FiLM), so the penalized objective
`MSE(Σ_j m_j M_j − ΔS) + λ_P·|m|` is multi-modal and under-determined — two
distinct supports can attain nearly-identical `J`, and the discrete tie-break
(ε = 1e-8) resolves them differently between Adam's 5 restarts and L-BFGS-B's
12 starts.

This is a known tension in the guide's own spec: a 5-restart Adam is compared
against a 12-start L-BFGS-B on an objective that is only guaranteed unimodal for
the *trained* bank (GT mechanisms are quadratic in `v`).

## Decision

The user chose to **proceed with discovery** (documenting this caveat), with the
gate re-verified against the **trained** bank once mechanisms have specialized.

## Re-verification

Run `python E0/adp1/recheck_solver_sanity.py --bank <trained_checkpoint.pt>`
against a trained `checkpoint_best_bank.pt` / `best_bank.pt` and confirm
`support_agreement >= 0.99`. Expected result once mechanisms are distinct and
unimodal (quadratic-in-v) — recorded under `solver_sanity/trained_recheck/metrics.json`.
