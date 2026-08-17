"""
DTSA's three transformation operators -- swap, shift, symmetry (Cinar et al. 2020 §3.1, p. 883).

WRITTEN FRESH, ON PURPOSE. `src/atsa_jssp/operators.py` is verified and is NOT imported here.

  swap, shift  -- identical in semantics to ATSA's, and verified to agree on the case both
                  papers demonstrate (see test_operators.py::test_swap_agrees_with_atsa and
                  ::test_shift_agrees_with_atsa). Rewritten anyway so that the DTSA
                  implementation is self-contained and can be audited without cross-reading
                  another algorithm's module.
  symmetry     -- MUST be written fresh. ATSA's `symmetry(v, r1, r2)` takes a pivot and a
                  half-width and can only ever produce the ADJACENT-block special case
                  (the DTSA specification §3 note S2). DTSA takes two independent block positions with a
                  common size. Reusing ATSA's would silently restrict DTSA to a subset of its
                  own operator -- and to the one subset, C4, that we are supposed to be TESTING
                  (the sampler design notes), not assuming.

All three are POSITION PERMUTATIONS: they rearrange elements and never change the multiset of
values. That is what makes the random-key representation admissible (the DTSA adaptation notes D1) and
it is asserted directly in the tests.

INDEXING: 0-indexed, matching the rest of the repo. The paper's figures are 1-indexed; every
docstring gives the figure's example in the paper's indexing and then in ours.

FLAGS. Every ambiguity in the DTSA specification §6 that touches an operator is a keyword flag whose
default is the most literal reading of the paper:

  symmetry_sampler      U1  -- how (a, b, L) are drawn.       default "C1" (maximally literal)
  shift_allow_x_gt_y    U3  -- whether shift accepts x > y.   default False (prose defines x < y)

The two remaining flags from §6 are algorithm-level, not operator-level, and live on
`dtsa_reference.Config`:  st_direction (U5), st_tie_break (U4).
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------------------------
# The operators
# ----------------------------------------------------------------------------------------------


def swap(v: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Fig. 2 (p. 883). Exchange the values at two positions. "Small changes on the tree."

        "Two different random numbers are created between 1 and N ... These two positions in
         the tree is swapped for creating new seed."

    Paper's example, 1-indexed:  swap([1 2 3 4 5 6], 2, 5) -> [1 5 3 4 2 6]
    Here,           0-indexed:   swap([1 2 3 4 5 6], 1, 4) -> [1 5 3 4 2 6]
    """
    out = v.copy()
    out[x], out[y] = out[y], out[x]
    return out


def shift(v: np.ndarray, x: int, y: int, *, allow_x_gt_y: bool = False) -> np.ndarray:
    """
    Fig. 3 (p. 883). "Medium changes on the tree."

        "Decision variable in the position of x, is memorized and other decision variables are
         shifted to the left in the range of [x,y]. After then the memorized decision variable
         is assigned to the position of y."

    i.e. a one-place left rotation of the closed interval [x, y].

    Paper's example, 1-indexed:  shift([1 2 3 4 5 6], 3, 5) -> [1 2 4 5 3 6]
    Here,           0-indexed:   shift([1 2 3 4 5 6], 2, 4) -> [1 2 4 5 3 6]

    U3: the prose defines only x < y, and Fig. 3 shows only x < y. `allow_x_gt_y` defaults to
    False, which REFUSES the undefined case rather than inventing a meaning for it. Setting it
    True adopts the mirror-image right shift -- which is what `src/atsa_jssp/operators.py` does
    for ATSA, and which DTSA never specifies. Keep the default unless testing U3.
    """
    if x == y:
        raise ValueError("shift: x and y must differ")
    if x > y and not allow_x_gt_y:
        raise ValueError(
            f"shift: x={x} > y={y} is undefined in DTSA (U3). The paper's prose defines only a "
            "left shift within [x, y]. Pass allow_x_gt_y=True to adopt the mirror-image right "
            "shift, which is ATSA's behaviour and is NOT specified by DTSA."
        )
    out = v.copy()
    t = out[x]
    if x < y:
        out[x:y] = out[x + 1:y + 1]
    else:                                     # only reachable with allow_x_gt_y=True
        out[y + 1:x + 1] = out[y:x]
    out[y] = t
    return out


def symmetry(v: np.ndarray, a: int, b: int, L: int) -> np.ndarray:
    """
    Fig. 4 (p. 883). "Big changes on the tree."

        "Two random positions for blocks which their block size is same random number of
         elements are determined. Each elements of blocks are inversed in their block and after
         then determined blocks are swapped."

    Block A = positions a .. a+L-1, block B = positions b .. b+L-1, same length L.
    Reverse each in place, then exchange the two blocks.

    Paper's example, 1-indexed:  symmetry([1 2 3 4 5 6], (2,3), (4,5)) -> [1 5 4 3 2 6]
    Here,           0-indexed:   symmetry([1 2 3 4 5 6], a=1, b=3, L=2) -> [1 5 4 3 2 6]

    ⚠️ THE PAPER'S EXAMPLE IS DEGENERATE and does not pin this operator down. Its blocks are
    ADJACENT (b == a+L), and for adjacent blocks reverse-each-then-swap collapses to reversing
    the single span [a, b+L-1] -- so Fig. 4 is equally consistent with a plain contiguous
    reversal. The two readings diverge only for non-adjacent blocks. See the DTSA specification §2.3, and
    test_operators.py::test_symmetry_non_adjacent_is_not_a_contiguous_reversal, which is the
    test that would have caught a wrong "same as ATSA" verdict.

    The operator is symmetric in its two blocks: symmetry(v, a, b, L) == symmetry(v, b, a, L).

    U2: overlapping blocks are REFUSED, not repaired. §3.1 claims no repair mechanism is needed;
    that is only true if the sampler never proposes an overlap, so the samplers below never do.
    """
    if L < 1:
        raise ValueError("symmetry: block size L must be >= 1")
    lo, hi = (a, b) if a <= b else (b, a)
    if lo < 0 or hi + L > v.shape[0]:
        raise ValueError(f"symmetry: blocks [{a},{a+L-1}] / [{b},{b+L-1}] fall outside 0..{v.shape[0]-1}")
    if hi - lo < L:
        raise ValueError(
            f"symmetry: blocks at {a} and {b} with L={L} overlap (U2). DTSA gives no repair "
            "rule; the samplers in this module never propose an overlap."
        )
    out = v.copy()
    block_a = v[a:a + L][::-1]        # "each elements of blocks are inversed in their block"
    block_b = v[b:b + L][::-1]
    out[a:a + L] = block_b            # "and after then determined blocks are swapped"
    out[b:b + L] = block_a
    return out


# ----------------------------------------------------------------------------------------------
# U1 -- the symmetry samplers. Definitions and rationale: the sampler design notes (pre-registered).
# Each returns (a, b, L), 0-indexed, blocks guaranteed disjoint and in bounds.
# ----------------------------------------------------------------------------------------------


def _uniform_disjoint_pair(D: int, L: int, rng: np.random.Generator) -> tuple[int, int]:
    """
    Uniform over UNORDERED disjoint block-position pairs for a given L.

    Equivalent in distribution to "draw a, b independently and resample while they overlap",
    which is how the sampler design notes C1 states it, but it terminates in O(1) instead of rejecting.
    (Independent draws conditioned on non-overlap are uniform over ordered disjoint pairs, which
    is 2-to-1 onto unordered pairs; and the operator is symmetric in its two blocks, so ordered
    and unordered give the same distribution over outcomes.)
    """
    # a in [0, D-2L], b in [a+L, D-L]; total pairs = (D-2L+1)(D-2L+2)/2
    m = D - 2 * L                      # >= 0 whenever L <= D//2
    # Pick a with weight proportional to the number of b's available: (m - a + 1).
    weights = np.arange(m + 1, 0, -1, dtype=np.float64)
    a = int(rng.choice(m + 1, p=weights / weights.sum()))
    b = int(rng.integers(a + L, D - L + 1))
    return a, b


def sample_C1(D: int, rng: np.random.Generator) -> tuple[int, int, int]:
    """C1, THE DEFAULT -- maximally literal: uniform L, uniform positions, no overlap."""
    L = int(rng.integers(1, D // 2 + 1))
    a, b = _uniform_disjoint_pair(D, L, rng)
    return a, b, L


def sample_C2(D: int, rng: np.random.Generator) -> tuple[int, int, int]:
    """C2 -- uniform over all feasible (a, b, L) triples; large L is penalised automatically."""
    Ls = np.arange(1, D // 2 + 1)
    counts = (D - 2 * Ls + 1) * (D - 2 * Ls + 2) / 2.0
    L = int(rng.choice(Ls, p=counts / counts.sum()))
    a, b = _uniform_disjoint_pair(D, L, rng)
    return a, b, L


def sample_C3(D: int, rng: np.random.Generator) -> tuple[int, int, int]:
    """C3 -- bounded block size, L <= ceil(D/10). The cap is OURS; the paper gives none."""
    cap = max(1, min(-(-D // 10), D // 2))
    L = int(rng.integers(1, cap + 1))
    a, b = _uniform_disjoint_pair(D, L, rng)
    return a, b, L


def sample_C4(D: int, rng: np.random.Generator) -> tuple[int, int, int]:
    """C4 -- adjacent blocks (b = a+L), which collapses to a contiguous reversal, i.e. a 2-opt
    move. Matches Fig. 4's example; contradicts the prose. See the sampler design notes."""
    L = int(rng.integers(1, D // 2 + 1))
    a = int(rng.integers(0, D - 2 * L + 1))
    return a, a + L, L


SYMMETRY_SAMPLERS = {"C1": sample_C1, "C2": sample_C2, "C3": sample_C3, "C4": sample_C4}


# ----------------------------------------------------------------------------------------------
# Position samplers for swap and shift. Both papers say only "two different random numbers
# between 1 and N", which is unambiguous.
# ----------------------------------------------------------------------------------------------


def sample_two_positions(D: int, rng: np.random.Generator) -> tuple[int, int]:
    """Two distinct positions in [0, D-1]."""
    x, y = rng.choice(D, size=2, replace=False)
    return int(x), int(y)


def sample_shift_positions(D: int, rng: np.random.Generator,
                           *, allow_x_gt_y: bool = False) -> tuple[int, int]:
    """Two distinct positions; ordered x < y unless U3's flag says otherwise."""
    x, y = sample_two_positions(D, rng)
    if not allow_x_gt_y and x > y:
        x, y = y, x
    return x, y


# ----------------------------------------------------------------------------------------------
# Apply one operator to a vector, drawing its own positions. Used by the DTSA kernel.
# ----------------------------------------------------------------------------------------------


def apply_operator(name: str, v: np.ndarray, rng: np.random.Generator, *,
                   symmetry_sampler: str = "C1",
                   shift_allow_x_gt_y: bool = False) -> np.ndarray:
    """Draw this operator's parameters and apply it. `name` in {"swap", "shift", "symmetry"}."""
    D = v.shape[0]
    if name == "swap":
        return swap(v, *sample_two_positions(D, rng))
    if name == "shift":
        return shift(v, *sample_shift_positions(D, rng, allow_x_gt_y=shift_allow_x_gt_y),
                     allow_x_gt_y=shift_allow_x_gt_y)
    if name == "symmetry":
        return symmetry(v, *SYMMETRY_SAMPLERS[symmetry_sampler](D, rng))
    raise ValueError(f"unknown operator {name!r}")


# Fig. 6 lines 16-21 / 24-29: the order is fixed and the operator is NOT sampled per seed --
# all three fire every time, one per seed slot, against each of two source trees. NS = 6 = 3 x 2.
OPERATOR_ORDER = ("swap", "shift", "symmetry")
