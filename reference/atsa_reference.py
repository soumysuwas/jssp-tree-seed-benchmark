"""
Reference ATSA implementation — a LITERAL transcription of Şahman (2022)
Algorithm 2 (Figure 6). Plain NumPy, no tricks, deliberately readable.

This is the ORACLE. Port it to numba for speed; keep this to diff against.
Every config flag corresponds to a documented ambiguity in
the ATSA design notes. Defaults = the most literal reading of the paper.

Run:  python reference/atsa_reference.py            # smoke test on ta01
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

from decoder_reference import Instance, evaluate
from operators_reference import (
    swap, symmetry, shift,
    rand_swap_positions, rand_symmetry_positions, rand_shift_positions,
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
    operator_space: str = "continuous"  # A1: "continuous" | "permutation"
    branch_granularity: str = "seed"    # A2: "seed" (Alg.2 literal) | "dimension" (Alg.1 style)
    strict_fe_cap: bool = False         # §5: False = literal (may overshoot)


# ----------------------------------------------------------------------------
def _st_fires(rnd: float, ST: float, sense: str) -> bool:
    """The ST test. Both readings exist in the paper; see 01_SPEC_ATSA.md §3."""
    return rnd < ST if sense == "rand_lt_st" else ST < rnd


def _eq3(tree, best, tree_r, rng, cfg):
    """Seed = Tree + r * (Best - Tree_r).  Exploitation."""
    r = rng.uniform(-1.0, 1.0, tree.shape[0])
    return np.clip(tree + r * (best - tree_r), cfg.dmin, cfg.dmax)


def _eq4(tree, best, tree_r, rng, cfg):
    """Seed = Tree + r * (Tree - Tree_r).  Exploration."""
    r = rng.uniform(-1.0, 1.0, tree.shape[0])
    return np.clip(tree + r * (tree - tree_r), cfg.dmin, cfg.dmax)


def _eq_per_dimension(tree, best, tree_r, rng, cfg, thresh):
    """branch_granularity='dimension': choose Eq.3/Eq.4 independently per dim."""
    D = tree.shape[0]
    r = rng.uniform(-1.0, 1.0, D)
    pick3 = rng.random(D) < thresh
    seed = np.where(pick3, tree + r * (best - tree_r), tree + r * (tree - tree_r))
    return np.clip(seed, cfg.dmin, cfg.dmax)


# ----------------------------------------------------------------------------
def atsa(inst: Instance, seed: int, cfg: Config = Config(), log_every: int | None = None):
    """
    Returns dict with best Cmax, FEs used, iterations, and the convergence trace.

    Algorithm 2 line-by-line. Line numbers in comments refer to
    the ATSA specification §4.
    """
    rng = np.random.default_rng(seed)
    D = inst.D
    N = cfg.N
    max_fes = D * cfg.fe_multiplier
    L = max(1, int(round(cfg.low_seed_frac * N)))
    U = max(L, int(round(cfg.high_seed_frac * N)))

    # ---- initialisation (Algorithm 1 lines 2-4): N trees, N FEs ----
    trees = rng.uniform(cfg.dmin, cfg.dmax, size=(N, D))
    fitness = np.array([evaluate(t, inst) for t in trees], dtype=np.int64)
    fes = N
    b = int(fitness.argmin())
    best, best_fit = trees[b].copy(), int(fitness[b])
    trace = [(fes, best_fit)]
    iters = 0

    while fes < max_fes:                                           # line 1
        iters += 1
        for i in range(N):                                         # line 2
            NS = int(rng.integers(L, U + 1))                       # line 3
            r_idx = int(rng.integers(0, N - 1))                    # line 3: r != i
            if r_idx >= i:
                r_idx += 1
            tree_r = trees[r_idx]
            tree_i = trees[i]

            cand_best, cand_best_fit = None, None

            for _ in range(NS):                                    # line 4
                if cfg.strict_fe_cap and fes >= max_fes:
                    break

                if rng.random() < 0.5:                             # line 5
                    # ---------- mutation-operator branch ----------
                    if _st_fires(rng.random(), cfg.ST, cfg.st_sense):   # line 6
                        # line 7-8: swap on BEST, 1 FE
                        p1, p2 = rand_swap_positions(D, rng)
                        s = swap(best, p1, p2)
                        f = evaluate(s, inst); fes += 1
                        cands = [(f, s)]
                    else:
                        # line 10-13: symmetry on NEIGHBOUR + shift on BEST, 2 FEs
                        q1, q2 = rand_symmetry_positions(D, rng)
                        sy = symmetry(tree_r, q1, q2)
                        t1, t2 = rand_shift_positions(D, rng)
                        sh = shift(best, t1, t2)
                        f_sy = evaluate(sy, inst)
                        f_sh = evaluate(sh, inst)
                        fes += 2
                        cands = [(f_sy, sy), (f_sh, sh)]
                        cands = [min(cands, key=lambda z: z[0])]   # line 13
                else:
                    # ---------- classic TSA equation branch ----------
                    if cfg.branch_granularity == "dimension":
                        s = _eq_per_dimension(tree_i, best, tree_r, rng, cfg, 0.75)
                    elif rng.random() < 0.75:                      # line 16
                        s = _eq3(tree_i, best, tree_r, rng, cfg)
                    else:                                          # line 19
                        s = _eq4(tree_i, best, tree_r, rng, cfg)
                    f = evaluate(s, inst); fes += 1                # lines 18/21
                    cands = [(f, s)]

                for f, s in cands:                                 # line 25
                    if cand_best_fit is None or f < cand_best_fit:
                        cand_best_fit, cand_best = f, s

            # line 26: greedy replacement
            if cand_best is not None and cand_best_fit < fitness[i]:
                trees[i] = cand_best
                fitness[i] = cand_best_fit
                if cand_best_fit < best_fit:                       # line 28
                    best_fit, best = cand_best_fit, cand_best.copy()

        trace.append((fes, best_fit))
        if log_every and iters % log_every == 0:
            print(f"    iter {iters:5d}  FEs {fes:>9,}/{max_fes:,}  best {best_fit}")

    return dict(cmax=best_fit, fes=fes, iters=iters, trace=trace,
                seed=seed, config=asdict(cfg))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, time, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from instance_reference import load_taillard_original

    inst = load_taillard_original(
        pathlib.Path(__file__).parents[1] / "data/raw/tai15_15.txt")[0]
    inst.name = "ta01"
    print(f"{inst.name}: n={inst.n} m={inst.m} D={inst.D} "
          f"MaxFEs={inst.D*1000:,}  known optimum=1231")

    cfg = Config(fe_multiplier=int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    t0 = time.time()
    res = atsa(inst, seed=0, cfg=cfg, log_every=200)
    dt = time.time() - t0
    print(f"\n  Cmax={res['cmax']}  FEs={res['fes']:,}  iters={res['iters']}  "
          f"time={dt:.1f}s")
    print(f"  paper ATSA ta01: mean 1444.8, med 1445, min 1347, max 1517")
