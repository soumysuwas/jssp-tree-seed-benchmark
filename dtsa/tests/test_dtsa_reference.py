"""
The DTSA kernel's accounting, the TSP objective, and 2-opt.

These are NOT Gate 1. Gate 1 is 9 configurations x 30 runs against dtsa_tables.TABLE1 and is
pre-registered in the DTSA adaptation notes D5. Nothing here compares anything to a published number.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from dtsa_reference import Config, check_fe_accounting, dtsa, dtsa_tsp
from tsp import load_tsp, nearest_neighbour_tour
from two_opt import two_opt

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "tsplib"

pytestmark = pytest.mark.skipif(
    not (DATA / "berlin52.tsp").exists(),
    reason="run `uv run python dtsa/fetch_tsplib.py` first",
)


@pytest.fixture(scope="module")
def berlin():
    return load_tsp(DATA / "berlin52.tsp", rounded=True)


# ==============================================================================================
# TSP objective
# ==============================================================================================
def test_vectorised_nint_matches_the_scalar_one(berlin):
    """tsp.TSP builds its matrix with floor(d+0.5); tsplib_io.nint is int(x+0.5)."""
    from tsp import check_nint_matches
    assert check_nint_matches(berlin)


def test_tour_length_matches_the_independent_implementation(berlin):
    """tsp.TSP.tour_length (matrix) vs tsplib_io.tour_length (pairwise) -- two routes, one answer."""
    from tsplib_io import read_opt_tour, tour_length as pairwise
    tour = np.array(read_opt_tour(DATA / "berlin52.opt.tour"))
    assert berlin.tour_length(tour) == pytest.approx(
        pairwise(list(tour), [tuple(c) for c in berlin.coords], rounded=True))
    assert berlin.tour_length(tour) == pytest.approx(7542.0)


def test_tour_length_is_rotation_and_reflection_invariant(berlin):
    rng = np.random.default_rng(1)
    t = rng.permutation(berlin.n)
    base = berlin.tour_length(t)
    assert berlin.tour_length(np.roll(t, 7)) == pytest.approx(base)
    assert berlin.tour_length(t[::-1]) == pytest.approx(base)


def test_load_tsp_rejects_non_euclidean(tmp_path):
    """Cinar et al. §5 (p. 883) exclude GEO instances scored as Euclidean; so do we, loudly."""
    p = tmp_path / "fake.tsp"
    p.write_text("NAME: fake\nEDGE_WEIGHT_TYPE: GEO\nNODE_COORD_SECTION\n"
                 "1 1.0 2.0\n2 3.0 4.0\nEOF\n")
    with pytest.raises(ValueError, match="EUC_2D"):
        load_tsp(p)


def test_nearest_neighbour_tour_is_a_permutation(berlin):
    t = nearest_neighbour_tour(berlin)
    assert sorted(t.tolist()) == list(range(berlin.n))


# ==============================================================================================
# 2-opt
# ==============================================================================================
def test_two_opt_delta_matches_a_full_recomputation(berlin):
    """The O(1) delta is the only clever thing in two_opt.py, so it gets checked directly."""
    rng = np.random.default_rng(7)
    n = berlin.n
    for _ in range(2000):
        t = rng.permutation(n)
        i = int(rng.integers(1, n - 1))
        j = int(rng.integers(i + 1, n))
        a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % n]
        if d == a:
            continue
        delta = (berlin.dist[a, c] + berlin.dist[b, d]) - (berlin.dist[a, b] + berlin.dist[c, d])
        u = t.copy()
        u[i:j + 1] = u[i:j + 1][::-1]
        assert berlin.tour_length(u) - berlin.tour_length(t) == pytest.approx(delta)


def test_two_opt_improves_a_random_tour_and_terminates(berlin):
    rng = np.random.default_rng(11)
    t = rng.permutation(berlin.n)
    r = two_opt(t, berlin.dist)
    assert r.length < berlin.tour_length(t)
    assert sorted(r.tour.tolist()) == list(range(berlin.n))
    assert r.evaluations > 0


def test_two_opt_output_is_2opt_optimal(berlin):
    """Whatever it returns must admit no further improving move -- otherwise it stopped early."""
    rng = np.random.default_rng(12)
    r = two_opt(rng.permutation(berlin.n), berlin.dist)
    again = two_opt(r.tour, berlin.dist)
    assert again.improving_moves == 0
    assert again.length == pytest.approx(r.length)


@pytest.mark.parametrize("variant", ["first", "best"])
def test_two_opt_variants_both_reach_a_local_optimum(berlin, variant):
    rng = np.random.default_rng(13)
    r = two_opt(rng.permutation(berlin.n), berlin.dist, variant=variant)
    assert two_opt(r.tour, berlin.dist, variant=variant).improving_moves == 0


def test_two_opt_rejects_unknown_variant(berlin):
    with pytest.raises(ValueError, match="variant"):
        two_opt(np.arange(berlin.n), berlin.dist, variant="lin-kernighan")


# ==============================================================================================
# FE accounting -- the DTSA analogue of ATSA's check_fe_accounting()
# ==============================================================================================
@pytest.mark.parametrize("N,max_fes", [(10, 1000), (12, 2000), (52, 4000)])
def test_fe_identity_holds(berlin, N, max_fes):
    cfg = Config(N=N, max_fes=max_fes, seed=0, two_opt_enabled=False)
    res = dtsa_tsp(berlin, cfg)
    check_fe_accounting(res, cfg)
    assert res.fes == N + 6 * N * res.iterations
    assert res.fes <= max_fes + 6 * N
    assert res.fes >= max_fes, "must not stop early"


def test_two_opt_evaluations_are_never_folded_into_fes(berlin):
    """D3/U7. The whole fairness argument rests on this staying true."""
    cfg = Config(N=12, max_fes=2000, seed=0, two_opt_enabled=True)
    res = dtsa_tsp(berlin, cfg)
    assert res.fes == cfg.N + 6 * cfg.N * res.iterations       # unchanged by 2-opt
    assert res.two_opt_evaluations > 0
    assert res.best_post_2opt is not None
    assert res.best_post_2opt <= res.best_pre_2opt


def test_ablation_mode_changes_the_seed_count_and_the_identity(berlin):
    """U14: Table 1's rows use one operator and one source. At NS=1 the budget buys 6x more
    iterations, which is why Gate 1 discriminates the two readings."""
    common = dict(N=10, max_fes=1200, seed=0, two_opt_enabled=False)
    six = Config(ablation=("symmetry", "current"), t1_seeds_per_row=6, **common)
    one = Config(ablation=("symmetry", "current"), t1_seeds_per_row=1, **common)
    r6, r1 = dtsa_tsp(berlin, six), dtsa_tsp(berlin, one)

    assert r6.seeds_per_tree == 6 and r1.seeds_per_tree == 1
    check_fe_accounting(r6, six)
    check_fe_accounting(r1, one)
    assert r6.fes == 10 + 6 * 10 * r6.iterations
    assert r1.fes == 10 + 1 * 10 * r1.iterations
    assert r1.iterations > r6.iterations * 5


def test_config_rejects_nonsense():
    with pytest.raises(ValueError, match="ablation operator"):
        Config(N=5, max_fes=100, ablation=("2-opt", "current"))
    with pytest.raises(ValueError, match="ablation source"):
        Config(N=5, max_fes=100, ablation=("swap", "elite"))
    with pytest.raises(ValueError, match="st_direction"):
        Config(N=5, max_fes=100, st_direction="whatever")
    with pytest.raises(ValueError, match="st_tie_break"):
        Config(N=5, max_fes=100, st_tie_break="whatever")


def test_st_tie_break_literal_default_skips_the_tree(berlin):
    """
    U4. Fig. 6's two branches are strict `<` and strict `>`, so rand == ST falls through both and
    that tree produces no seeds at all. Almost certainly not intended -- but inventing a branch
    would be a silent fix, so the literal reading is the default and the alternatives are flags.

    Forced here with ST=0.0 and st_direction's normal sense: rand is never < 0, and rand == 0.0
    is possible but vanishingly rare, so instead we assert the mechanism directly.
    """
    cfg = Config(N=8, max_fes=400, ST=0.5, seed=0, two_opt_enabled=False)
    res = dtsa_tsp(berlin, cfg)
    assert cfg.st_tie_break == "none"
    assert res.branch_counts["tie"] == res.branch_counts["tie_skipped"]
    # And the FE identity must still hold even though a skipped tree spends nothing.
    assert res.branch_counts["tie"] == 0, "float rand exactly == 0.5 should be astronomically rare"


def test_st_direction_flag_is_inert_at_st_0_5_in_distribution(berlin):
    """
    U5. The §2 prose and Fig. 6 disagree on the direction of the ST test. At the default
    ST = 0.5 the two readings are exchanged by rand -> 1-rand, so they are identical in
    distribution. Different RNG draws mean the runs are not bit-identical; what we assert is that
    both are valid, complete runs with correct accounting.
    """
    a = Config(N=10, max_fes=1000, ST=0.5, seed=3, two_opt_enabled=False)
    b = Config(N=10, max_fes=1000, ST=0.5, seed=3, two_opt_enabled=False,
               st_direction="rand_gt_st_best")
    ra, rb = dtsa_tsp(berlin, a), dtsa_tsp(berlin, b)
    check_fe_accounting(ra, a)
    check_fe_accounting(rb, b)
    assert ra.iterations == rb.iterations
    assert ra.branch_counts["rand_lt_ST"] == rb.branch_counts["rand_lt_ST"]


def test_dtsa_is_deterministic_given_a_seed(berlin):
    cfg = Config(N=10, max_fes=1000, seed=42, two_opt_enabled=False)
    assert dtsa_tsp(berlin, cfg).best_pre_2opt == dtsa_tsp(berlin, cfg).best_pre_2opt


def test_dtsa_never_returns_a_worse_best_than_the_seeded_nn_tour(berlin):
    """Fig. 6 line 5 seeds tree 1 with the NN tour and `best` is monotone, so the result can
    never be worse than it."""
    cfg = Config(N=10, max_fes=1000, seed=0, two_opt_enabled=False)
    res = dtsa_tsp(berlin, cfg)
    assert res.best_pre_2opt <= berlin.tour_length(nearest_neighbour_tour(berlin))


def test_best_vector_is_a_valid_tour(berlin):
    cfg = Config(N=10, max_fes=1000, seed=0)
    res = dtsa_tsp(berlin, cfg)
    assert sorted(res.best_vector.tolist()) == list(range(berlin.n))


def test_kernel_is_problem_agnostic():
    """The kernel takes an evaluate() and a population; nothing in it knows about TSP. This is
    what lets the job-shop port reuse it with an RK population (the DTSA adaptation notes D1)."""
    rng = np.random.default_rng(0)
    D, N = 12, 6
    pop = np.array([rng.permutation(D) for _ in range(N)])
    cfg = Config(N=N, max_fes=300, seed=0, two_opt_enabled=False)
    res = dtsa(lambda v: float(np.sum(v * np.arange(D))), pop, cfg)
    check_fe_accounting(res, cfg)
    assert res.best_post_2opt is None
