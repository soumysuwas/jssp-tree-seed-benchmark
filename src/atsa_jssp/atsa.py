"""
ATSA — Advanced Tree-Seed Algorithm. Sahman (2022) Algorithm 2 / Figure 6.

A LITERAL transcription. Ported line-for-line from reference/atsa_reference.py; the
Algorithm-2 line numbers from the ATSA specification §4 are kept as comments. The branch
structure is deliberately NOT "cleaned up" — its literalness is the whole point of a
reproduction. Every config flag corresponds to a documented ambiguity in
the ATSA design notes; defaults = the most literal reading of the paper.

Structure mirrors decoder.py: `_atsa_kernel` is the njit hot loop, `atsa()` is the thin
Python wrapper that returns the same dict shape as the reference oracle.

FE ACCOUNTING (the ATSA specification §5) — the thing most likely to break silently:
  - initialisation costs N = 40 FEs
  - branch B (symmetry-on-neighbour + shift-on-best) costs 2 FEs for ONE seed
  - every other branch costs 1 FE
  - E[FE/seed] = 0.1(1) + 0.4(2) + 0.375(1) + 0.125(1) = 1.4
  - `FEs < maxFEs` is checked at the while (line 1) ONLY. Overshoot up to N*U*2 = 800
    is expected and correct. We do not break out of the seed loop.
The kernel returns the branch counters so a caller can assert all of this instead of
trusting it; `check_fe_accounting()` does exactly that.

RNG NOTE: the kernel uses numba's np.random (Mersenne Twister), not NumPy's PCG64
Generator, which numba cannot use in nopython mode. Streams therefore differ from the
reference oracle, so runs are NOT bit-comparable to it — but each run is seeded and
exactly reproducible. Per the design notes §4 exact mean-matching is not the
success criterion (the paper is MATLAB Mersenne anyway); the distribution is what matters.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from atsa_jssp._compat import njit  # optional-Numba shim (pure-Python fallback)

from atsa_jssp.decoder import Instance, evaluate_fast
from atsa_jssp.operators import (
    swap_nb, symmetry_nb, shift_nb,
    rand_two_positions_nb, rand_symmetry_positions_nb,
)


# ----------------------------------------------------------------------------
@dataclass
class Config:
    # --- Table 4 of the paper. DO NOT TUNE. ---
    N: int = 40                     # stand size
    ST: float = 0.2                 # search tendency
    low_seed_frac: float = 0.10     # L = 0.1*N
    high_seed_frac: float = 0.25    # U = 0.25*N
    dmin: float = -5.0
    dmax: float = 5.0
    fe_multiplier: int = 1000       # MaxFEs = D * 1000

    # --- ambiguity switches (see the ATSA design notes) ---
    st_sense: str = "rand_lt_st"        # Q1: "rand_lt_st" (pseudocode) | "st_lt_rand" (prose)
    operator_space: str = "continuous"  # A1: "continuous" | "permutation"  -- INERT, see below
    branch_granularity: str = "seed"    # A2: "seed" (Alg.2 literal) | "dimension" (Alg.1 style)
    strict_fe_cap: bool = False         # §5: False = literal (may overshoot)
    operator_anchor: str = "literal"    # A3: "literal" = branches A/B seed from Best/Tree_r
                                        #     (Algorithm 2 lines 7/10/11, THE DEFAULT)
                                        #     "parent" = seed from tree_i, as basic TSA does.
                                        # A3 was never testable before; this flag tests it.
                                        # A flag that closes the gap is a finding about the
                                        # paper being underspecified, NOT a licence to switch.

    def limits(self) -> tuple[int, int]:
        """(L, U) — min/max seeds per tree."""
        L = max(1, int(round(self.low_seed_frac * self.N)))
        U = max(L, int(round(self.high_seed_frac * self.N)))
        return L, U


# `operator_space` is accepted, recorded in the CSV, and swept by
# the design notes §5 — but it provably cannot change a result, so it is not
# passed to the kernel. The reference oracle declares it and never reads it, for the
# same reason.
#
# WHY IT IS INERT: all three operators PERMUTE POSITIONS, and RK ranking is equivariant
# under position permutation. If x' = P(x) for a permutation P, then tau' = P(tau), hence
# seq' = (tau' mod n) = P(seq). So "apply the operator to the continuous vector, then
# decode" and "decode, then apply the operator to the permutation" yield the IDENTICAL job
# sequence — for swap, symmetry AND shift. Verified empirically: 0 mismatches in 20,000
# random vectors x 3 operators on a 15x15 instance.
#
# CONSEQUENCE for the §5 sweep: its 8 combinations are only 4 DISTINCT configs. The
# continuous/permutation pairs are duplicates and their results will differ by nothing but
# the RNG stream. the ATSA design notes A1 calls this "low risk either way"; it is in
# fact zero risk, and A1 can be closed rather than swept.


# ----------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _atsa_kernel(route, ptime, n, m, N, ST, dmin, dmax, L, U, max_fes,
                 st_rand_lt_st, branch_dim, strict_cap, anchor_parent, seed,
                 trace_fes, trace_fit, best_out,
                 instrument, i_prod, i_fes, i_beat, i_best, div_it, div_dist, div_nuniq,
                 init_pop, use_init_pop):
    """
    Algorithm 2, verbatim. Returns
      (best_fit, fes, iters, n_trace, n_seeds, cA, cB, cC, cD, cE, best_branch, n_div)
    where cA..cD are the branch counters from the ATSA specification §4.1 and cE counts
    branch_granularity="dimension" seeds, which have no single C/D label (each dimension
    picks Eq.3/Eq.4 independently) but still cost exactly 1 FE.

    Branch index order everywhere: 0=A_swap 1=B_sym_shift 2=C_eq3 3=D_eq4 4=E_eq_perdim.

    `instrument` gates the Task-2 diagnostics (per-branch acceptance + diversity sampling).
    Off by default. When off the only cost is a handful of predictable branches per seed;
    the diversity computation (O(N^2 * D)) is skipped entirely.
    """
    np.random.seed(seed)
    D = n * m

    # ---- initialisation (Algorithm 1 lines 2-4): N trees, N FEs ----
    if use_init_pop:
        trees = init_pop.copy()
    else:
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
    cA = 0; cB = 0; cC = 0; cD = 0; cE = 0
    best_branch = -1
    n_div = 0

    # scratch buffers — reused, so the hot loop allocates nothing
    s1 = np.empty(D, dtype=np.float64)
    s2 = np.empty(D, dtype=np.float64)
    cand = np.empty(D, dtype=np.float64)

    while fes < max_fes:                                           # line 1
        iters += 1
        for i in range(N):                                         # line 2
            NS = np.random.randint(L, U + 1)                       # line 3
            r_idx = np.random.randint(0, N - 1)                    # line 3: r != i
            if r_idx >= i:
                r_idx += 1
            tree_r = trees[r_idx]
            tree_i = trees[i]

            has_cand = False
            cand_fit = 0
            cand_branch = -1

            for _ in range(NS):                                    # line 4
                if strict_cap and fes >= max_fes:
                    break

                if np.random.random() < 0.5:                       # line 5
                    # ---------- mutation-operator branch ----------
                    rnd = np.random.random()                       # line 6: independent draw
                    if (rnd < ST) if st_rand_lt_st else (ST < rnd):
                        # line 7-8: swap on BEST, 1 FE.
                        # A3/operator_anchor: "literal" anchors on Best (as printed);
                        # "parent" anchors on tree_i (as basic TSA does).
                        p1, p2 = rand_two_positions_nb(D)
                        if anchor_parent:
                            swap_nb(tree_i, p1, p2, s1)
                        else:
                            swap_nb(best, p1, p2, s1)
                        f = evaluate_fast(s1, route, ptime, n, m)
                        fes += 1
                        cA += 1
                        if instrument:
                            i_prod[0] += 1
                            i_fes[0] += 1
                            if f < fitness[i]:
                                i_beat[0] += 1
                        if not has_cand or f < cand_fit:           # line 25
                            for k in range(D):
                                cand[k] = s1[k]
                            cand_fit = f
                            has_cand = True
                            cand_branch = 0
                    else:
                        # line 10-13: symmetry on NEIGHBOUR + shift on BEST, 2 FEs
                        q1, q2 = rand_symmetry_positions_nb(D)
                        t1, t2 = rand_two_positions_nb(D)
                        if anchor_parent:
                            symmetry_nb(tree_i, q1, q2, s1)
                            shift_nb(tree_i, t1, t2, s2)
                        else:
                            symmetry_nb(tree_r, q1, q2, s1)
                            shift_nb(best, t1, t2, s2)
                        f_sy = evaluate_fast(s1, route, ptime, n, m)
                        f_sh = evaluate_fast(s2, route, ptime, n, m)
                        fes += 2
                        cB += 1
                        if instrument:
                            i_prod[1] += 1
                            i_fes[1] += 2
                            if (f_sy if f_sy <= f_sh else f_sh) < fitness[i]:
                                i_beat[1] += 1
                        # line 13: select the best seed between Sy and Sh (ties -> Sy,
                        # matching the oracle's min(..., key=) first-wins)
                        if f_sy <= f_sh:
                            f = f_sy
                            if not has_cand or f < cand_fit:       # line 25
                                for k in range(D):
                                    cand[k] = s1[k]
                                cand_fit = f
                                has_cand = True
                                cand_branch = 1
                        else:
                            f = f_sh
                            if not has_cand or f < cand_fit:       # line 25
                                for k in range(D):
                                    cand[k] = s2[k]
                                cand_fit = f
                                has_cand = True
                                cand_branch = 1
                else:
                    # ---------- classic TSA equation branch ----------
                    used_eq3 = False
                    if branch_dim:
                        # A2 "dimension": choose Eq.3/Eq.4 independently per dim
                        for k in range(D):
                            rk = np.random.uniform(-1.0, 1.0)
                            if np.random.random() < 0.75:
                                val = tree_i[k] + rk * (best[k] - tree_r[k])
                            else:
                                val = tree_i[k] + rk * (tree_i[k] - tree_r[k])
                            if val < dmin:
                                val = dmin
                            elif val > dmax:
                                val = dmax
                            s1[k] = val
                        cE += 1
                    elif np.random.random() < 0.75:                # line 16
                        # Eq.3: Seed = Tree + r*(Best - Tree_r).  Exploitation.
                        for k in range(D):                         # line 17: clamp
                            rk = np.random.uniform(-1.0, 1.0)
                            val = tree_i[k] + rk * (best[k] - tree_r[k])
                            if val < dmin:
                                val = dmin
                            elif val > dmax:
                                val = dmax
                            s1[k] = val
                        cC += 1
                        used_eq3 = True
                    else:                                          # line 19
                        # Eq.4: Seed = Tree + r*(Tree - Tree_r).  Exploration.
                        for k in range(D):                         # line 20: clamp
                            rk = np.random.uniform(-1.0, 1.0)
                            val = tree_i[k] + rk * (tree_i[k] - tree_r[k])
                            if val < dmin:
                                val = dmin
                            elif val > dmax:
                                val = dmax
                            s1[k] = val
                        cD += 1
                    f = evaluate_fast(s1, route, ptime, n, m)      # lines 18/21
                    fes += 1
                    if instrument:
                        bi = 4 if branch_dim else (2 if used_eq3 else 3)
                        i_prod[bi] += 1
                        i_fes[bi] += 1
                        if f < fitness[i]:
                            i_beat[bi] += 1
                    if not has_cand or f < cand_fit:               # line 25
                        for k in range(D):
                            cand[k] = s1[k]
                        cand_fit = f
                        has_cand = True
                        cand_branch = 4 if branch_dim else (2 if used_eq3 else 3)

                n_seeds += 1

            # line 26: greedy replacement
            if has_cand and cand_fit < fitness[i]:
                for k in range(D):
                    trees[i, k] = cand[k]
                fitness[i] = cand_fit
                if cand_fit < best_fit:                            # line 28
                    best_fit = cand_fit
                    for k in range(D):
                        best[k] = cand[k]
                    best_branch = cand_branch
                    if instrument and cand_branch >= 0:
                        i_best[cand_branch] += 1

        trace_fes[n_trace] = fes
        trace_fit[n_trace] = best_fit
        n_trace += 1

        # ---- Task 2(b): population diversity, sampled every 50 iterations ----
        if instrument and (iters % 50 == 1) and n_div < div_it.shape[0]:
            tot = 0.0
            for a in range(N):
                for b2 in range(a + 1, N):
                    acc = 0.0
                    for k in range(D):
                        d = trees[a, k] - trees[b2, k]
                        acc += d * d
                    tot += np.sqrt(acc)
            div_it[n_div] = iters
            div_dist[n_div] = tot / (N * (N - 1) / 2.0)       # mean pairwise Euclidean
            nu = 0
            for a in range(N):
                seen = False
                for b2 in range(a):
                    if fitness[b2] == fitness[a]:
                        seen = True
                        break
                if not seen:
                    nu += 1
            div_nuniq[n_div] = nu                             # distinct fitness values
            n_div += 1

    for k in range(D):
        best_out[k] = best[k]
    return (best_fit, fes, iters, n_trace, n_seeds, cA, cB, cC, cD, cE,
            best_branch, n_div)


# ----------------------------------------------------------------------------
BRANCH_NAMES = ["A_swap", "B_sym_shift", "C_eq3", "D_eq4", "E_eq_perdim"]


def atsa(inst: Instance, seed: int, cfg: Config = Config(),
         instrument: bool = False, init_pop: np.ndarray | None = None) -> dict:
    """
    Run ATSA on `inst` with the given RNG seed. Returns a dict with the best Cmax,
    FEs used, iterations, the convergence trace, and the branch counters.

    `instrument=True` additionally records per-branch acceptance and population
    diversity (Task 2). Off by default — it costs an O(N^2 * D) diversity sample every
    50 iterations.

    `init_pop` optionally supplies the initial population, shape (cfg.N, inst.D), in
    place of the internal `np.random.uniform(dmin, dmax, ...)`. It is used verbatim
    (copied, not clipped) and still costs the same N initial FEs, so the budget is
    unchanged. When None — the default — the RNG stream and every result are
    bit-identical to before this parameter existed.
    """
    route, ptime = inst.arrays()
    D = inst.D
    max_fes = D * cfg.fe_multiplier
    L, U = cfg.limits()

    # Cheapest possible FE/iteration is N*L*1 (every seed a 1-FE branch), so this
    # bounds the number of while-iterations. +2 for the initial row and slack.
    max_iters = max_fes // (cfg.N * L) + 2
    trace_fes = np.zeros(max_iters + 1, dtype=np.int64)
    trace_fit = np.zeros(max_iters + 1, dtype=np.int64)
    best_out = np.zeros(D, dtype=np.float64)

    n_div_max = max_iters // 50 + 2 if instrument else 1
    i_prod = np.zeros(5, dtype=np.int64)
    i_fes = np.zeros(5, dtype=np.int64)
    i_beat = np.zeros(5, dtype=np.int64)
    i_best = np.zeros(5, dtype=np.int64)
    div_it = np.zeros(n_div_max, dtype=np.int64)
    div_dist = np.zeros(n_div_max, dtype=np.float64)
    div_nuniq = np.zeros(n_div_max, dtype=np.int64)

    use_init_pop = init_pop is not None
    if use_init_pop:
        if init_pop.shape != (cfg.N, D):
            raise ValueError(f"init_pop must have shape {(cfg.N, D)}, got {init_pop.shape}")
        pop0 = np.ascontiguousarray(init_pop, dtype=np.float64)
    else:
        pop0 = np.empty((1, 1), dtype=np.float64)

    (best_fit, fes, iters, n_trace, n_seeds,
     cA, cB, cC, cD, cE, best_branch, n_div) = _atsa_kernel(
        route, ptime, inst.n, inst.m, cfg.N, cfg.ST, cfg.dmin, cfg.dmax,
        L, U, max_fes,
        cfg.st_sense == "rand_lt_st",
        cfg.branch_granularity == "dimension",
        bool(cfg.strict_fe_cap),
        cfg.operator_anchor == "parent",
        seed, trace_fes, trace_fit, best_out,
        bool(instrument), i_prod, i_fes, i_beat, i_best, div_it, div_dist, div_nuniq,
        pop0, use_init_pop,
    )

    out = dict(
        cmax=int(best_fit), fes=int(fes), iters=int(iters),
        trace=list(zip(trace_fes[:n_trace].tolist(), trace_fit[:n_trace].tolist())),
        best=best_out, seed=seed, config=asdict(cfg),
        n_seeds=int(n_seeds),
        branch_counts=dict(A_swap=int(cA), B_sym_shift=int(cB), C_eq3=int(cC),
                           D_eq4=int(cD), E_eq_perdim=int(cE)),
    )
    if instrument:
        out["instrumentation"] = dict(
            produced={BRANCH_NAMES[k]: int(i_prod[k]) for k in range(5)},
            fes_spent={BRANCH_NAMES[k]: int(i_fes[k]) for k in range(5)},
            beat_parent={BRANCH_NAMES[k]: int(i_beat[k]) for k in range(5)},
            became_best={BRANCH_NAMES[k]: int(i_best[k]) for k in range(5)},
            acceptance={BRANCH_NAMES[k]: (i_beat[k] / i_prod[k] if i_prod[k] else float("nan"))
                        for k in range(5)},
            final_best_branch=(BRANCH_NAMES[best_branch] if best_branch >= 0 else "initialisation"),
            diversity=dict(
                iters=div_it[:n_div].tolist(),
                mean_pairwise_dist=div_dist[:n_div].tolist(),
                distinct_fitness=div_nuniq[:n_div].tolist(),
            ),
        )
    return out


def check_fe_accounting(res: dict, cfg: Config = Config(), algorithm: str = "ATSA") -> dict:
    """
    Verify the FE bookkeeping of a finished run against the ATSA specification §5.
    Returns the measured quantities; raises AssertionError if a branch is mis-wired.

    Not decorative: if E[FE/seed] drifts off target, branch B is being counted as 1 FE (or
    not firing at its rate), and every downstream number is wrong.

    ⚠️ The target is NOT always 1.4. the ATSA specification §5 states E[FE/seed] = 1.4 as if
    universal, but that figure assumes st_sense="rand_lt_st" (the default). Branch B is the
    only 2-FE branch, so E[FE/seed] = 1 + P(B), and Q1's flag flips P(B):
        rand_lt_st (pseudocode): P(B) = 0.5*(1-ST) = 0.40  -> E = 1.40
        st_lt_rand (prose):      P(B) = 0.5*ST     = 0.10  -> E = 1.10
    Hard-coding 1.4 would make the the design notes §5 sweep un-runnable — the
    st_lt_rand arm would abort on a correct implementation. So derive it from cfg.
    """
    bc = res["branch_counts"]
    n_seeds = res["n_seeds"]
    seed_fes = res["fes"] - cfg.N                       # init costs N (A7)
    # branch B is the only 2-FE branch; every other branch (incl. the per-dimension
    # variant E) costs exactly 1. E[FE/seed] = 1.4 either way.
    expected = (bc["A_swap"] + 2 * bc["B_sym_shift"] + bc["C_eq3"] + bc["D_eq4"]
                + bc["E_eq_perdim"])
    assert sum(bc.values()) == n_seeds, \
        f"branch counts {sum(bc.values())} != seeds produced {n_seeds}"
    assert expected == seed_fes, \
        f"FEs implied by branch counts ({expected}) != FEs counted ({seed_fes}) — a branch is mis-wired"
    fe_per_seed = seed_fes / n_seeds
    if algorithm.upper() == "TSA":
        target = 1.0          # Algorithm 1 has no 2-FE branch: every seed costs exactly 1
    else:
        p_b = 0.5 * (1.0 - cfg.ST) if cfg.st_sense == "rand_lt_st" else 0.5 * cfg.ST
        target = 1.0 + p_b                              # 1.4 at the defaults
    assert target - 0.05 <= fe_per_seed <= target + 0.05, \
        (f"E[FE/seed] = {fe_per_seed:.4f}, expected ~{target:.2f} for "
         f"st_sense={cfg.st_sense} ST={cfg.ST} (the ATSA specification §5)")
    return dict(
        fe_per_seed=fe_per_seed, fe_per_seed_target=target,
        branch_frac={k: v / n_seeds for k, v in bc.items()},
        seed_fes=seed_fes, n_seeds=n_seeds,
    )
