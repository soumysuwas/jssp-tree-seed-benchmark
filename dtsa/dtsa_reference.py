"""
DTSA reference implementation -- Fig. 6 (journal p. 884) transcribed line for line.

Same discipline as `reference/atsa_reference.py`: the figure's line numbers are carried as
comments, the structure follows the pseudocode even where that is not how one would write it
from scratch, and NOTHING is optimised. Readability is the deliverable. A fast version, if it is
ever needed, gets written separately and validated against this one.

Every ambiguity in the DTSA specification §6 is a flag on `Config`, defaulting to the most literal reading:

  st_direction        U5   rand < ST -> best        (Fig. 6 line 15 and Fig. 1's ST box)
  st_tie_break        U4   rand == ST -> no seeds   (both comparisons are strict in Fig. 6)
  symmetry_sampler    U1   "C1"                     (the sampler design notes, pre-registered)
  shift_allow_x_gt_y  U3   False                    (the prose defines only x < y)
  t1_seeds_per_row    U14  6                        (Fig. 6 line 12, even in ablation mode)
  two_opt_enabled          True                     (Fig. 6 line 38 -- the literal algorithm)

FE ACCOUNTING is asserted on every run, the DTSA analogue of ATSA's `check_fe_accounting()`.
That assert caught two real bugs on the ATSA side (the project design notes §9 items 5-6):

    fes == N + seeds_per_tree * N * iterations        exactly
    fes <= max_fes + seeds_per_tree * N               overshoot bounded by one iteration

2-opt's evaluations are counted SEPARATELY and are never folded into `fes` (D3, U7). The paper
does fold them in -- by never counting them at all -- which is exactly the problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import operators as ops


# ==============================================================================================
@dataclass
class Config:
    """Table 4 of the DTSA paper is not a thing; parameters come from §5.2 and per-experiment."""

    N: int                                    # Fig. 6 line 3  -- stand size
    max_fes: int                              # Fig. 6 line 1
    ST: float = 0.5                           # Fig. 6 line 4  -- §5.2, p. 884
    NS: int = 6                               # Fig. 6 line 12 -- §5.2: fixed at 6

    # --- ambiguity flags, all defaulting to the literal reading -------------------------------
    st_direction: str = "rand_lt_st_best"     # U5; alt "rand_gt_st_best" (§2 prose)
    st_tie_break: str = "none"                # U4; alt "best" / "current"
    symmetry_sampler: str = "C1"              # U1
    shift_allow_x_gt_y: bool = False          # U3

    # --- Table 1 ablation mode (one operator, one source tree) --------------------------------
    ablation: tuple[str, str] | None = None   # e.g. ("symmetry", "current")
    t1_seeds_per_row: int = 6                 # U14: 6 (literal) or 1
    ablation_split_random: bool = False       # U17; default False = every seed from the named
    #                                           source. True = Fig. 6's structure kept, i.e.
    #                                           half the seeds still come from Tree(k).
    best_update: str = "deferred"             # F6; "deferred" is LITERAL (Fig. 6 lines 35-36,
    #                                           after the for-loop). "immediate" updates `best`
    #                                           as soon as a tree improves -- NOT more faithful,
    #                                           and it stays off. Diagnostic arm only.

    # --- local search (D3) --------------------------------------------------------------------
    two_opt_enabled: bool = True              # Fig. 6 line 38
    two_opt_variant: str = "first"            # U6; ours, the paper says nothing

    seed: int = 0

    def __post_init__(self) -> None:
        if self.ablation is not None:
            op, src = self.ablation
            if op not in ops.OPERATOR_ORDER:
                raise ValueError(f"ablation operator {op!r} not in {ops.OPERATOR_ORDER}")
            if src not in ("current", "random", "best"):
                raise ValueError(f"ablation source {src!r} not in current/random/best")
        if self.st_direction not in ("rand_lt_st_best", "rand_gt_st_best"):
            raise ValueError(f"bad st_direction {self.st_direction!r}")
        if self.st_tie_break not in ("none", "best", "current"):
            raise ValueError(f"bad st_tie_break {self.st_tie_break!r}")
        if self.best_update not in ("deferred", "immediate"):
            raise ValueError(f"bad best_update {self.best_update!r}")

    @property
    def seeds_per_tree(self) -> int:
        """Fig. 6 line 31 increments fes by exactly this much, per tree, per iteration."""
        return self.t1_seeds_per_row if self.ablation is not None else self.NS


@dataclass
class Result:
    best_vector: np.ndarray
    best_pre_2opt: float                      # DTSA-core  -- Table 1's "TSA Mean" column
    best_post_2opt: float | None              # DTSA+LS    -- Table 1's "TSA + 2Opt Mean" column
    fes: int                                  # search only; 2-opt NEVER included
    iterations: int
    seeds_per_tree: int
    two_opt_evaluations: int = 0              # reported separately, per D3
    two_opt_moves: int = 0
    branch_counts: dict = field(default_factory=dict)   # how often each ST branch fired


# ==============================================================================================
def check_fe_accounting(res: Result, cfg: Config) -> None:
    """The identity that must hold on every run. Loud failure, not a warning."""
    expected = cfg.N + res.seeds_per_tree * cfg.N * res.iterations
    assert res.fes == expected, (
        f"FE accounting broken: fes={res.fes}, expected "
        f"N + seeds_per_tree*N*iterations = {cfg.N} + {res.seeds_per_tree}*{cfg.N}*"
        f"{res.iterations} = {expected}"
    )
    bound = cfg.max_fes + res.seeds_per_tree * cfg.N
    assert res.fes <= bound, (
        f"FE overshoot too large: fes={res.fes} > max_fes + seeds_per_tree*N = {bound}. "
        "Termination is checked only at the while (Fig. 6 line 10), so at most one iteration "
        "may overshoot."
    )


# ==============================================================================================
def dtsa(evaluate: Callable[[np.ndarray], float],
         initial_population: np.ndarray,
         cfg: Config,
         local_search: Callable[[np.ndarray], tuple[np.ndarray, float, int, int]] | None = None,
         ) -> Result:
    """
    Fig. 6, line by line.

    `initial_population` is (N, D) and is built by the caller, because line 5's "nearest neighbor
    tour" is problem-specific (the DTSA adaptation notes D2). Lines 5 and 6 are therefore the caller's
    responsibility; everything from line 7 down is here.

    `local_search` implements line 38 and returns (tour, length, evaluations, improving_moves).
    Its evaluations are reported separately and never added to `fes`.
    """
    rng = np.random.default_rng(cfg.seed)

    trees = np.array(initial_population, copy=True)
    N, D = trees.shape
    if N != cfg.N:
        raise ValueError(f"initial_population has {N} trees, Config.N is {cfg.N}")

    # ---- line 7: Calculate the objective functions of all trees ------------------------------
    fitness = np.array([evaluate(t) for t in trees], dtype=np.float64)

    # ---- line 8: Set function evaluation number (fes) as N ------------------------------------
    fes = N

    # ---- line 9: Determine the best tree via using the objective function values (best) -------
    b = int(np.argmin(fitness))
    best_vector = trees[b].copy()
    best_fitness = float(fitness[b])

    seeds_per_tree = cfg.seeds_per_tree
    iterations = 0
    branch_counts = {"rand_lt_ST": 0, "rand_gt_ST": 0, "tie": 0, "tie_skipped": 0}

    # ---- line 10: While fes is smaller than maxfes --------------------------------------------
    while fes < cfg.max_fes:
        iterations += 1

        # ---- line 11: For all trees (Tree(1) to Tree(N)) --------------------------------------
        for i in range(N):
            # ---- line 12: Determine the number of seed (ns) as 6 ------------------------------
            ns = seeds_per_tree

            # ---- line 13: Determine a random tree (except the current tree) (Tree(k)) ---------
            # Drawn ONCE per tree per iteration, before the ST test -- the DTSA specification F3.
            k = int(rng.integers(0, N - 1))
            if k >= i:
                k += 1

            # ---- line 14: Create a random number between 0 and 1 (rand) ----------------------
            rand = float(rng.random())

            # ---- lines 15 / 23: which tree seeds s1..s3 --------------------------------------
            # U5: Fig. 6 line 15 says rand < ST -> best; the §2 prose says the opposite. At the
            # default ST=0.5 the two are exchanged by rand -> 1-rand and are indistinguishable.
            lt_gives_best = cfg.st_direction == "rand_lt_st_best"
            if rand < cfg.ST:
                branch_counts["rand_lt_ST"] += 1
                primary = best_vector if lt_gives_best else trees[i]
            elif rand > cfg.ST:
                branch_counts["rand_gt_ST"] += 1
                primary = trees[i] if lt_gives_best else best_vector
            else:
                # ---- U4: rand == ST is covered by NEITHER branch. Fig. 6 uses strict < and >. -
                # The literal default produces no seeds at all for this tree, which is almost
                # certainly not what the authors intended -- but inventing a branch would be a
                # silent fix, so it is a flag instead.
                branch_counts["tie"] += 1
                if cfg.st_tie_break == "none":
                    branch_counts["tie_skipped"] += 1
                    continue
                primary = best_vector if cfg.st_tie_break == "best" else trees[i]

            # ---- lines 16-21 / 24-29: create the seeds ---------------------------------------
            # NS = 6 is structurally 3 operators x 2 source trees. The operator is NOT sampled
            # per seed: all three fire every time, in the fixed order swap, shift, symmetry,
            # against each source -- the DTSA specification F1.
            seeds = []
            if cfg.ablation is None:
                for source in (primary, trees[k]):
                    for name in ops.OPERATOR_ORDER:
                        seeds.append(ops.apply_operator(
                            name, source, rng,
                            symmetry_sampler=cfg.symmetry_sampler,
                            shift_allow_x_gt_y=cfg.shift_allow_x_gt_y))
            else:
                # Table 1 mode: one operator (U14 sets ns).
                op_name, src_name = cfg.ablation
                source = {"current": trees[i], "random": trees[k], "best": best_vector}[src_name]
                if not cfg.ablation_split_random:
                    # U17 default, literal to the row label: every seed from the named tree.
                    sources = [source] * ns
                else:
                    # U17 alternative: Fig. 6's structure is kept and only the PRIMARY source
                    # varies -- lines 19-21/27-29 always seed from Tree(k), so half the seeds
                    # still come from the random tree. A "random tree" row is then simply the
                    # case where the primary source is Tree(k) as well.
                    half = ns // 2
                    sources = [source] * (ns - half) + [trees[k]] * half
                for src_vec in sources:
                    seeds.append(ops.apply_operator(
                        op_name, src_vec, rng,
                        symmetry_sampler=cfg.symmetry_sampler,
                        shift_allow_x_gt_y=cfg.shift_allow_x_gt_y))

            assert len(seeds) == ns, f"produced {len(seeds)} seeds, ns={ns}"

            # ---- line 31: Calculate the objective function of seeds ... increase fes by 6 -----
            seed_fitness = [evaluate(s) for s in seeds]
            fes += ns

            # ---- line 32: Determine the best seed from the ns seeds (bestseed) ---------------
            j = int(np.argmin(seed_fitness))

            # ---- line 33: If the bestseed is smaller than current tree, replace --------------
            if seed_fitness[j] < fitness[i]:
                trees[i] = seeds[j]
                fitness[i] = seed_fitness[j]
                # F6 diagnostic arm only. The literal algorithm defers this to lines 35-36.
                if cfg.best_update == "immediate" and fitness[i] < best_fitness:
                    best_fitness = float(fitness[i])
                    best_vector = trees[i].copy()

        # ---- line 35: Determine the best tree (tempbest) -------------------------------------
        # Note this happens AFTER the whole for-loop, so the `best` used for seeding above is one
        # iteration stale -- the DTSA specification F6. That is what the figure says; keep it.
        t = int(np.argmin(fitness))

        # ---- line 36: If the tempbest is smaller than best, replace --------------------------
        if fitness[t] < best_fitness:
            best_fitness = float(fitness[t])
            best_vector = trees[t].copy()

    # ---- end while (line 37) -----------------------------------------------------------------
    res = Result(
        best_vector=best_vector,
        best_pre_2opt=best_fitness,
        best_post_2opt=None,
        fes=fes,
        iterations=iterations,
        seeds_per_tree=seeds_per_tree,
        branch_counts=branch_counts,
    )
    check_fe_accounting(res, cfg)

    # ---- line 38: Apply the 2-opt algorithm using the best individual -------------------------
    # Outside the loop, once, on `best` only. Its evaluations are NOT added to fes -- the paper
    # does not count them either, which is the defect (U7); we count and report them instead.
    if cfg.two_opt_enabled and local_search is not None:
        tour, length, evals, moves = local_search(best_vector)
        res.best_vector = tour
        res.best_post_2opt = length
        res.two_opt_evaluations = evals
        res.two_opt_moves = moves

    # ---- line 39: Report the best ------------------------------------------------------------
    return res


# ==============================================================================================
# Thin TSP adapter -- lines 5 and 6, plus the line-38 local search. Kept separate so that the
# kernel above stays problem-agnostic and can take a job-shop population unchanged.
# ==============================================================================================
def dtsa_tsp(problem, cfg: Config, *, nn_start: int = 0, nn_best_start: bool = False) -> Result:
    """
    Build the population per Fig. 6 lines 5-6 and run DTSA on a TSP instance.

    `nn_best_start` is the U12 diagnostic arm and is NOT a default: Fig. 6 line 5 says "nearest
    neighbor tour", singular, so best-of-n is better-performing rather than more faithful.
    """
    from tsp import best_nearest_neighbour_tour, nearest_neighbour_tour
    from two_opt import two_opt

    rng = np.random.default_rng(cfg.seed)
    D = problem.D

    # ---- line 5: Determine the first tree as nearest neighbor tour ---------------------------
    pop = np.empty((cfg.N, D), dtype=np.int64)
    pop[0] = (best_nearest_neighbour_tour(problem)[0] if nn_best_start
              else nearest_neighbour_tour(problem, start=nn_start))

    # ---- line 6: Create all trees (except the first one) with random permutations ------------
    for i in range(1, cfg.N):
        pop[i] = rng.permutation(D)

    def local_search(tour):
        r = two_opt(tour, problem.dist, variant=cfg.two_opt_variant)
        return r.tour, r.length, r.evaluations, r.improving_moves

    return dtsa(problem.tour_length, pop, cfg, local_search=local_search)
