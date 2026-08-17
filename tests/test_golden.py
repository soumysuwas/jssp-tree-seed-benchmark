"""
Golden tests. These encode every known-answer fact we have about the paper.
If any of these go red, STOP — nothing downstream is trustworthy.

Run:  uv run pytest -q
      uv run pytest -q -m "not slow"     # skip CP-SAT
"""
from __future__ import annotations
import pathlib
import numpy as np
import pytest

from decoder_reference import (
    Instance, rk_ranks, rk_to_job_sequence, build_schedule, evaluate,
    paper_2x3_instance,
)
from operators_reference import (
    swap, symmetry, shift,
    rand_swap_positions, rand_symmetry_positions, rand_shift_positions,
)
from instance_reference import (
    load_taillard_original, load_jsplib, TAILLARD_FILES, PAPER_INSTANCES, NOT_IN_PAPER,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ===========================================================================
# Layer 1 — the decoder. Known answers from the paper's own worked example.
# ===========================================================================
PAPER_X = np.array([-2.1, 4.5, 4.9, 1.2, -4.3, 3.8])


def test_paper_2x3_route_reindexing():
    """Table 1's times are indexed by MACHINE; the order is a separate column."""
    inst = paper_2x3_instance()
    assert inst.ptime.tolist() == [[8, 3, 6], [10, 4, 8]]
    assert (inst.route + 1).tolist() == [[1, 3, 2], [2, 1, 3]]


def test_paper_2x3_ranks():
    """Table 2 'Integer Series (phi)' row."""
    assert rk_ranks(PAPER_X).tolist() == [2, 5, 6, 3, 1, 4]


def test_paper_2x3_eq2():
    """Table 2 'Job Indexes' row. Eq.2: S = (tau mod n) + 1."""
    assert (rk_to_job_sequence(PAPER_X, n=2) + 1).tolist() == [1, 2, 1, 2, 2, 1]


def test_paper_2x3_gantt():
    """Figure 1's bars, exactly."""
    inst = paper_2x3_instance()
    seq = rk_to_job_sequence(PAPER_X, 2)
    _, starts, ends = build_schedule(seq, inst)
    assert starts.tolist() == [[0, 8, 11], [0, 10, 14]]
    assert ends.tolist() == [[8, 11, 17], [10, 14, 22]]


def test_paper_2x3_cmax_is_22_not_24():
    """
    PAPER DEFECT P1. The body text says 24. Figure 1's own title says
    'Makespan=22'. 22 is arithmetically correct. Assert 22.
    """
    inst = paper_2x3_instance()
    assert evaluate(PAPER_X, inst) == 22


@pytest.mark.parametrize("n,m", [(2, 3), (5, 4), (15, 15), (100, 20)])
def test_rk_always_feasible(n, m):
    """
    Eq.2 yields exactly m copies of every job for ANY real vector, because tau is
    a permutation of 1..n*m and residues mod n hit each class exactly m times.
    This is why no repair mechanism is ever needed.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        seq = rk_to_job_sequence(rng.uniform(-5, 5, n * m), n)
        assert np.array_equal(np.bincount(seq, minlength=n), np.full(n, m))


def test_schedule_respects_constraints():
    """No machine overlap; route order honoured. Belt-and-braces on the decoder."""
    rng = np.random.default_rng(7)
    inst = load_taillard_original(ROOT / "data/raw/tai15_15.txt")[0]
    for _ in range(50):
        seq = rk_to_job_sequence(rng.uniform(-5, 5, inst.D), inst.n)
        _, starts, ends = build_schedule(seq, inst)
        # route order
        for i in range(inst.n):
            for j in range(1, inst.m):
                assert starts[i, j] >= ends[i, j - 1]
        # machine exclusivity
        for mach in range(inst.m):
            iv = sorted((starts[i, j], ends[i, j])
                        for i in range(inst.n) for j in range(inst.m)
                        if inst.route[i, j] == mach)
            for (_, e1), (s2, _) in zip(iv, iv[1:]):
                assert s2 >= e1


# ===========================================================================
# Layer 2 — operators. Known answers from Figures 3, 4, 5.
# ===========================================================================
V = np.array([1, 2, 3, 4, 5, 6, 7])


def test_swap_figure3():
    assert swap(V, 2, 5).tolist() == [1, 2, 6, 4, 5, 3, 7]      # paper r1=3, r2=6


def test_symmetry_figure4():
    assert symmetry(V, 3, 2).tolist() == [1, 6, 5, 4, 3, 2, 7]  # paper r1=4, r2=2


def test_shift_figure5():
    assert shift(V, 1, 4).tolist() == [1, 3, 4, 5, 2, 6, 7]     # paper r1=2, r2=5


@pytest.mark.parametrize("D", [7, 225, 2000])
def test_operators_pure_and_measure_preserving(D):
    """
    Operators permute values -> the multiset is preserved -> a mutated vector is
    still inside [-5,5]. Consequence: clamp ONLY after Eq.3/Eq.4, never after a
    mutation.
    """
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = rng.uniform(-5, 5, D)
        snap = v.copy()
        for fn, pos in ((swap, rand_swap_positions),
                        (symmetry, rand_symmetry_positions),
                        (shift, rand_shift_positions)):
            a, b = pos(D, rng)
            out = fn(v, a, b)
            assert np.array_equal(v, snap), f"{fn.__name__} mutated its input"
            assert np.allclose(np.sort(out), np.sort(v))
            assert out.min() >= -5 and out.max() <= 5


# ===========================================================================
# Layer 0 — data integrity.
# ===========================================================================
def test_all_80_instances_agree():
    """
    Taillard-original and JSPLIB must parse to identical Instances.
    This is the proof that the supplied data == the standard benchmark.

    Skips if the JSPLIB cross-check copies aren't present — run
    `scripts/fetch_data.py` once (needs network) to enable it. The cross-check
    is a nice-to-have proof, not a dependency: don't let it block the sprint.
    """
    if not (ROOT / "data/jsplib/ta01").exists():
        pytest.skip("data/jsplib/ missing — run scripts/fetch_data.py to enable the cross-check")
    checked = 0
    for fname, (lo, hi) in TAILLARD_FILES.items():
        p = ROOT / "data/raw" / fname
        if not p.exists():
            pytest.skip(f"{fname} missing — copy the supplied files into data/raw/ (data/README.md)")
        for k, a in enumerate(load_taillard_original(p)):
            q = ROOT / "data/jsplib" / f"ta{lo + k:02d}"
            if not q.exists():
                continue
            b = load_jsplib(q)
            assert (a.n, a.m) == (b.n, b.m)
            assert np.array_equal(a.route, b.route), f"ta{lo+k:02d} route mismatch"
            assert np.array_equal(a.ptime, b.ptime), f"ta{lo+k:02d} ptime mismatch"
            checked += 1
    assert checked == 80, f"only checked {checked}/80"


def test_paper_scope_is_40_instances_ending_at_ta75():
    """
    The paper's Table 3 uses 40 instances; the highest is ta75. the supplied data
    holds all 80 (it's the full Taillard benchmark) but the other 40 have no
    Table 5 row to compare against. Running them is wasted compute -- and
    ta76-80 are 100x20, the most expensive size there is.
    """
    assert len(PAPER_INSTANCES) == 40
    assert PAPER_INSTANCES[0] == "ta01"
    assert PAPER_INSTANCES[-1] == "ta75"
    assert "ta80" not in PAPER_INSTANCES
    assert "ta76" not in PAPER_INSTANCES
    assert len(NOT_IN_PAPER) == 40
    for base in (1, 11, 21, 31, 41, 51, 61, 71):
        for k in range(base, base + 5):
            assert f"ta{k:02d}" in PAPER_INSTANCES


def test_ta01_header_matches_taillard():
    """the supplied file is Taillard's ta01. Spot-check the first job."""
    inst = load_taillard_original(ROOT / "data/raw/tai15_15.txt")[0]
    assert (inst.n, inst.m, inst.D) == (15, 15, 225)
    assert (inst.route[0][:3] + 1).tolist() == [7, 13, 5]
    assert inst.ptime[0][:3].tolist() == [94, 66, 10]


# ===========================================================================
# Layer 3 — the njit port must equal the oracle. Enable once decoder.py exists.
# ===========================================================================
@pytest.mark.skipif(
    not (ROOT / "src/atsa_jssp/decoder.py").exists(),
    reason="requires src/atsa_jssp/decoder.py",
)
def test_njit_matches_oracle():
    from atsa_jssp.decoder import evaluate_fast
    rng = np.random.default_rng(3)
    for fname in ("tai15_15.txt",):
        for inst in load_taillard_original(ROOT / "data/raw" / fname)[:3]:
            r = inst.route.astype(np.int64)
            p = inst.ptime.astype(np.int64)
            for _ in range(10_000):
                x = rng.uniform(-5, 5, inst.D)
                assert evaluate_fast(x, r, p, inst.n, inst.m) == evaluate(x, inst)
