"""
GATE 0 -- the DTSA operators, before anything is allowed to run.

Four groups, per the DTSA adaptation notes D5:
  1. the three worked examples from Figs. 2-4, exactly as the DTSA specification §2 states them
  2. a NON-ADJACENT symmetry case, constructed by us, that discriminates the two-block operator
     from a contiguous reversal -- the test that would have caught a wrong "SAME" verdict
  3. every operator preserves the multiset of values (they are position permutations)
  4. D1 equivariance: applying an operator to the random-key vector and decoding gives the same
     job sequence as decoding and then applying it

Group 4 is the load-bearing one. If it fails, the DTSA adaptation notes D1's default is wrong and the
whole representation decision has to be reopened.
"""
from __future__ import annotations

import numpy as np
import pytest

import operators as ops
from operators import shift, swap, symmetry

# Read-only imports from the verified ATSA library (src/ is on pythonpath via pyproject).
from atsa_jssp.decoder import rk_to_job_sequence
from atsa_jssp.instance import load_ta

SEED = 20260723


# ==============================================================================================
# 1. The paper's worked examples (the DTSA specification §2). 1-indexed in the figures, 0-indexed here.
# ==============================================================================================
def test_fig2_swap():
    """Fig. 2, p. 883:  swap([1..6], 2, 5) -> [1 5 3 4 2 6]   (1-indexed)"""
    v = np.array([1, 2, 3, 4, 5, 6])
    assert list(swap(v, 1, 4)) == [1, 5, 3, 4, 2, 6]


def test_fig3_shift():
    """Fig. 3, p. 883:  shift([1..6], 3, 5) -> [1 2 4 5 3 6]   (1-indexed)"""
    v = np.array([1, 2, 3, 4, 5, 6])
    assert list(shift(v, 2, 4)) == [1, 2, 4, 5, 3, 6]


def test_fig4_symmetry():
    """Fig. 4, p. 883:  symmetry([1..6], (2,3), (4,5)) -> [1 5 4 3 2 6]   (1-indexed)

    Blocks 2-3 and 4-5, size 2. Reverse each -> [3 2] and [5 4]; swap them -> [1 5 4 3 2 6].
    """
    v = np.array([1, 2, 3, 4, 5, 6])
    assert list(symmetry(v, a=1, b=3, L=2)) == [1, 5, 4, 3, 2, 6]


def test_operators_do_not_mutate_their_input():
    v = np.array([1, 2, 3, 4, 5, 6])
    before = v.copy()
    swap(v, 1, 4), shift(v, 2, 4), symmetry(v, a=1, b=3, L=2)
    assert list(v) == list(before)


def test_swap_agrees_with_atsa_on_the_demonstrated_case():
    """swap is rewritten in dtsa/ rather than imported; check it still matches the verified one."""
    from atsa_jssp.operators import swap as atsa_swap
    v = np.arange(1, 8)
    assert list(swap(v, 2, 5)) == list(atsa_swap(v, 2, 5)) == [1, 2, 6, 4, 5, 3, 7]


def test_shift_agrees_with_atsa_on_the_demonstrated_case():
    from atsa_jssp.operators import shift as atsa_shift
    v = np.arange(1, 8)
    assert list(shift(v, 1, 4)) == list(atsa_shift(v, 1, 4)) == [1, 3, 4, 5, 2, 6, 7]


# ==============================================================================================
# 2. OUR construction -- the discriminating case the paper never shows
# ==============================================================================================
def test_symmetry_non_adjacent_is_not_a_contiguous_reversal():
    """
    ⚠️ THIS CASE IS OURS, NOT THE PAPER'S. Derived from the §3.1 prose ("two random positions
    for blocks"), which places no adjacency requirement on the blocks.

    Fig. 4's blocks are ADJACENT, and for adjacent blocks reverse-each-then-swap collapses to
    reversing the whole span -- so the paper's only worked example cannot tell the two-block
    operator apart from a contiguous reversal, and cannot tell DTSA's symmetry apart from ATSA's
    (the DTSA specification §3 note S2). A non-adjacent case can.

    Blocks A = positions 0-1 = [1 2], B = positions 4-5 = [5 6], L = 2.
      reverse each -> [2 1] and [6 5];  swap  -> [6 5 3 4 2 1]
      a contiguous reversal of 0..5 would give [6 5 4 3 2 1]  -- different at positions 2,3.
    """
    v = np.array([1, 2, 3, 4, 5, 6])
    got = list(symmetry(v, a=0, b=4, L=2))
    assert got == [6, 5, 3, 4, 2, 1]
    assert got != list(v[::-1]), "must not equal the contiguous reversal of the whole span"

    # And it is not equal to ANY single contiguous reversal of v.
    all_reversals = []
    for lo in range(6):
        for hi in range(lo + 1, 6):
            w = v.copy()
            w[lo:hi + 1] = w[lo:hi + 1][::-1]
            all_reversals.append(list(w))
    assert got not in all_reversals


def test_symmetry_adjacent_blocks_do_collapse_to_a_reversal():
    """The converse, which is why Fig. 4 is degenerate: with b == a+L the operator IS a
    contiguous reversal. This is candidate C4 in the sampler design notes."""
    rng = np.random.default_rng(SEED)
    for _ in range(500):
        D = int(rng.integers(4, 30))
        L = int(rng.integers(1, D // 2 + 1))
        a = int(rng.integers(0, D - 2 * L + 1))
        v = rng.permutation(D)
        expect = v.copy()
        expect[a:a + 2 * L] = expect[a:a + 2 * L][::-1]
        assert list(symmetry(v, a, a + L, L)) == list(expect)


def test_symmetry_is_symmetric_in_its_two_blocks():
    rng = np.random.default_rng(SEED)
    for _ in range(500):
        D = int(rng.integers(4, 40))
        L = int(rng.integers(1, D // 2 + 1))
        a, b = ops._uniform_disjoint_pair(D, L, rng)
        v = rng.permutation(D)
        assert list(symmetry(v, a, b, L)) == list(symmetry(v, b, a, L))


def test_symmetry_refuses_overlapping_blocks():
    """U2 -- no repair rule exists, so overlap is refused rather than silently corrupted."""
    v = np.arange(10)
    with pytest.raises(ValueError, match="overlap"):
        symmetry(v, a=0, b=2, L=3)


def test_shift_refuses_x_greater_than_y_by_default():
    """U3 -- the prose defines only x < y. The undefined case is refused, not invented."""
    v = np.arange(10)
    with pytest.raises(ValueError, match="U3"):
        shift(v, 7, 2)
    assert list(shift(v, 7, 2, allow_x_gt_y=True)) == [0, 1, 7, 2, 3, 4, 5, 6, 8, 9]


# ==============================================================================================
# 3. All three are position permutations
# ==============================================================================================
@pytest.mark.parametrize("sampler", ["C1", "C2", "C3", "C4"])
def test_operators_preserve_the_value_multiset(sampler):
    """>= 10,000 random vectors per operator. This property is what licenses the random-key
    representation (the DTSA adaptation notes D1) and it is why no clamping is ever needed."""
    rng = np.random.default_rng(SEED)
    n_each = 3400                                   # x 3 operators > 10,000
    for _ in range(n_each):
        D = int(rng.integers(4, 60))
        v = rng.uniform(-5, 5, size=D)
        before = np.sort(v)
        for name in ops.OPERATOR_ORDER:
            out = ops.apply_operator(name, v, rng, symmetry_sampler=sampler)
            assert out.shape == v.shape
            assert np.array_equal(np.sort(out), before)


@pytest.mark.parametrize("sampler", ["C1", "C2", "C3", "C4"])
def test_symmetry_samplers_emit_only_legal_disjoint_blocks(sampler):
    rng = np.random.default_rng(SEED)
    for _ in range(4000):
        D = int(rng.integers(4, 80))
        a, b, L = ops.SYMMETRY_SAMPLERS[sampler](D, rng)
        lo, hi = min(a, b), max(a, b)
        assert L >= 1
        assert lo >= 0 and hi + L <= D
        assert hi - lo >= L, "blocks must be disjoint (U2)"
        if sampler == "C4":
            assert hi - lo == L, "C4 blocks are adjacent by construction"


def test_c3_respects_its_block_size_cap():
    rng = np.random.default_rng(SEED)
    D = 200
    sizes = [ops.sample_C3(D, rng)[2] for _ in range(2000)]
    assert max(sizes) <= -(-D // 10)


# ==============================================================================================
# 4. D1 EQUIVARIANCE -- the load-bearing test
# ==============================================================================================
EQUIVARIANCE_INSTANCES = ["ta01", "ta11", "ta21"]
EQUIVARIANCE_TRIALS = 1200          # x 3 operators x 3 instances > 10,000; see metrics.py


@pytest.mark.parametrize("inst_name", EQUIVARIANCE_INSTANCES)
def test_d1_equivariance_rk_vs_sequence(inst_name):
    """
    the DTSA adaptation notes D1. All three DTSA operators are position permutations, and random-key
    ranking is equivariant under position permutation (the project design notes §8 D4). Therefore

        rk_to_job_sequence(op(x), n) == op(rk_to_job_sequence(x, n))

    i.e. mutating the random-key vector and then decoding gives EXACTLY the job sequence you get
    by decoding first and mutating after. That equality is what lets DTSA run on the RK vector
    and reuse the verified ATSA decoder unchanged.

    IF THIS FAILS, D1's default is wrong, the representation decision must be reopened, and
    nothing downstream should be run. Report immediately.
    """
    inst = load_ta(inst_name)
    n, D = inst.n, inst.D
    rng = np.random.default_rng(SEED)

    for _ in range(EQUIVARIANCE_TRIALS):
        x = rng.uniform(-5, 5, size=D)
        seq = rk_to_job_sequence(x, n)
        for name in ops.OPERATOR_ORDER:
            # Same draw for both sides: apply the operator to the RK vector and to the decoded
            # sequence using identical parameters, by seeding two generators identically.
            r1 = np.random.default_rng(12345)
            r2 = np.random.default_rng(12345)
            mutated_rk = ops.apply_operator(name, x, r1)
            mutated_seq = ops.apply_operator(name, seq, r2)
            assert np.array_equal(rk_to_job_sequence(mutated_rk, n), mutated_seq), (
                f"{inst_name}/{name}: RK and sequence representations diverged -- "
                "the DTSA adaptation notes D1 is invalid"
            )


def test_d1_equivariance_holds_for_every_symmetry_sampler():
    """The equivariance argument is about position permutations, so it must not depend on which
    U1 candidate is in force. Checked so that adopting a different sampler after Gate 1 cannot
    quietly invalidate D1."""
    inst = load_ta("ta01")
    n, D = inst.n, inst.D
    rng = np.random.default_rng(SEED)
    for sampler in ("C1", "C2", "C3", "C4"):
        for _ in range(600):
            x = rng.uniform(-5, 5, size=D)
            seq = rk_to_job_sequence(x, n)
            r1 = np.random.default_rng(999)
            r2 = np.random.default_rng(999)
            mx = ops.apply_operator("symmetry", x, r1, symmetry_sampler=sampler)
            ms = ops.apply_operator("symmetry", seq, r2, symmetry_sampler=sampler)
            assert np.array_equal(rk_to_job_sequence(mx, n), ms), sampler
