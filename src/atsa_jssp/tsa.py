"""
Basic TSA — Tree-Seed Algorithm. Sahman (2022) Algorithm 1 / Figure 2, the baseline ATSA
must beat by ~30% (the paper claims 30.33%).

Spec: the ATSA specification §3. Same parameters as ATSA (Table 4): N=40, ST=0.2,
NS in [L,U]=[4,10], MaxFEs=D*1000, range [-5,5].

WHY THIS FILE EXISTS (beyond the baseline): it bisects the size-dependent drift.
TSA shares EVERYTHING with ATSA except the mutation operators — same decoder, same RK
encoding, same Eq.3/Eq.4, same FE accounting, same harness, same seeds. So:
    TSA drifts with D too  -> cause is in the SHARED core (decoder / RK / FE budget)
    TSA does not drift     -> cause is ATSA-specific (branches, operators, Best-anchoring)

Two structural differences from ATSA, both straight from the paper:
  1. Algorithm 1 has an explicit `for k=1 to D` loop with the ST test INSIDE it, so the
     Eq.3/Eq.4 choice is PER-DIMENSION. Algorithm 2 has no such loop (the design notes A2).
  2. Every seed is anchored on the parent `Tree_i`. ATSA's branches A/B instead seed from
     `Best`/`Tree_r` (the design notes A3).
Every seed costs exactly 1 FE, so E[FE/seed] = 1.0 (ATSA's is 1 + P(B)).
"""
from __future__ import annotations
from dataclasses import asdict
import numpy as np
from atsa_jssp._compat import njit  # optional-Numba shim (pure-Python fallback)

from atsa_jssp.atsa import Config
from atsa_jssp.decoder import Instance, evaluate_fast


@njit(cache=True, nogil=True)
def _tsa_kernel(route, ptime, n, m, N, ST, dmin, dmax, L, U, max_fes,
                st_rand_lt_st, strict_cap, seed, trace_fes, trace_fit, best_out):
    """
    Algorithm 1, verbatim. Returns (best_fit, fes, iters, n_trace, n_seeds).
    Line numbers below refer to the ATSA specification §3 / Figure 2.
    """
    np.random.seed(seed)
    D = n * m

    # ---- initialisation (lines 2-4): N trees, N FEs ----
    trees = np.random.uniform(dmin, dmax, (N, D))
    fitness = np.empty(N, dtype=np.int64)
    for i in range(N):
        fitness[i] = evaluate_fast(trees[i], route, ptime, n, m)
    fes = N
    b = 0
    for i in range(1, N):
        if fitness[i] < fitness[b]:
            b = i
    best = trees[b].copy()
    best_fit = fitness[b]

    trace_fes[0] = fes
    trace_fit[0] = best_fit
    n_trace = 1
    iters = 0
    n_seeds = 0

    s1 = np.empty(D, dtype=np.float64)
    cand = np.empty(D, dtype=np.float64)

    while fes < max_fes:                                   # line 5
        iters += 1
        for i in range(N):                                 # line 6
            NS = np.random.randint(L, U + 1)               # line 7
            r_idx = np.random.randint(0, N - 1)            # line 7: r != i
            if r_idx >= i:
                r_idx += 1
            tree_r = trees[r_idx]
            tree_i = trees[i]

            has_cand = False
            cand_fit = 0

            for _ in range(NS):                            # line 8
                if strict_cap and fes >= max_fes:
                    break
                for k in range(D):                         # line 9: PER-DIMENSION
                    rk = np.random.uniform(-1.0, 1.0)
                    rnd = np.random.random()
                    use_eq3 = (rnd < ST) if st_rand_lt_st else (ST < rnd)   # line 10
                    if use_eq3:
                        val = tree_i[k] + rk * (best[k] - tree_r[k])        # Eq.3
                    else:
                        val = tree_i[k] + rk * (tree_i[k] - tree_r[k])      # Eq.4
                    if val < dmin:                         # boundary check
                        val = dmin
                    elif val > dmax:
                        val = dmax
                    s1[k] = val
                f = evaluate_fast(s1, route, ptime, n, m)  # 1 FE per seed, always
                fes += 1
                n_seeds += 1
                if not has_cand or f < cand_fit:           # determine best seed
                    for k in range(D):
                        cand[k] = s1[k]
                    cand_fit = f
                    has_cand = True

            if has_cand and cand_fit < fitness[i]:         # greedy replacement
                for k in range(D):
                    trees[i, k] = cand[k]
                fitness[i] = cand_fit
                if cand_fit < best_fit:
                    best_fit = cand_fit
                    for k in range(D):
                        best[k] = cand[k]

        trace_fes[n_trace] = fes
        trace_fit[n_trace] = best_fit
        n_trace += 1

    for k in range(D):
        best_out[k] = best[k]
    return best_fit, fes, iters, n_trace, n_seeds


def tsa(inst: Instance, seed: int, cfg: Config = Config()) -> dict:
    """Run basic TSA. Returns the same dict shape as `atsa()` so the harness is shared."""
    route, ptime = inst.arrays()
    D = inst.D
    max_fes = D * cfg.fe_multiplier
    L, U = cfg.limits()

    max_iters = max_fes // (cfg.N * L) + 2
    trace_fes = np.zeros(max_iters + 1, dtype=np.int64)
    trace_fit = np.zeros(max_iters + 1, dtype=np.int64)
    best_out = np.zeros(D, dtype=np.float64)

    best_fit, fes, iters, n_trace, n_seeds = _tsa_kernel(
        route, ptime, inst.n, inst.m, cfg.N, cfg.ST, cfg.dmin, cfg.dmax,
        L, U, max_fes, cfg.st_sense == "rand_lt_st", bool(cfg.strict_fe_cap),
        seed, trace_fes, trace_fit, best_out,
    )

    return dict(
        cmax=int(best_fit), fes=int(fes), iters=int(iters),
        trace=list(zip(trace_fes[:n_trace].tolist(), trace_fit[:n_trace].tolist())),
        best=best_out, seed=seed, config=asdict(cfg),
        n_seeds=int(n_seeds),
        # TSA has no branches; every seed is 1 FE. Keys kept so the CSV schema is shared
        # with ATSA — C_eq3/D_eq4 are per-DIMENSION choices here, not per-seed, so they
        # are not counted: labelling them would imply a per-seed branch that does not exist.
        branch_counts=dict(A_swap=0, B_sym_shift=0, C_eq3=0, D_eq4=0, E_eq_perdim=int(n_seeds)),
    )
