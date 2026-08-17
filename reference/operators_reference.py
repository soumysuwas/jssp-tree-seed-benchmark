"""
Reference mutation operators for ATSA — swap, symmetry, shift.

VERIFIED against Şahman (2022) Figures 3, 4, 5 (and cross-checked against the
DTSA paper, Cinar et al. 2020, Figures 2-4, which define the same operators).

All operators are PURE: they return a new array, never mutate the input.
They operate on the CONTINUOUS vector (see the ATSA specification A1).

Run:  python reference/operators_reference.py
"""
from __future__ import annotations
import numpy as np


def swap(v: np.ndarray, r1: int, r2: int) -> np.ndarray:
    """Figure 3. Exchange the values at two positions. Smallest perturbation."""
    out = v.copy()
    out[r1], out[r2] = out[r2], out[r1]
    return out


def symmetry(v: np.ndarray, r1: int, r2: int) -> np.ndarray:
    """
    Figure 4. r1 = pivot position, r2 = block size.

    Two blocks of size r2 FLANK the pivot:
        left  = positions [r1-r2, r1-1]
        right = positions [r1+1, r1+r2]
    Each block is reversed, then the two blocks are swapped (DTSA §3.1).
    The pivot value itself does not move.

    That composite is exactly equivalent to reversing the whole span
    [r1-r2, r1+r2] (length 2*r2+1, centred on r1) — the pivot sits at the
    centre of an odd-length reversal so it stays put. We implement the
    equivalent one-liner.

    Paper example (1-indexed r1=4, r2=2):
        1 2 3 4 5 6 7  ->  1 6 5 4 3 2 7
    Here (0-indexed) r1=3, r2=2 -> reverse span [1, 5] inclusive.

    Largest perturbation of the three (DTSA §5.1 found symmetry produces the
    most diversified solutions, which is why it lands in ATSA's 40% branch).
    """
    lo, hi = r1 - r2, r1 + r2
    assert 0 <= lo and hi < v.shape[0], f"symmetry span [{lo},{hi}] out of bounds"
    out = v.copy()
    out[lo:hi + 1] = out[lo:hi + 1][::-1]
    return out


def shift(v: np.ndarray, r1: int, r2: int) -> np.ndarray:
    """
    Figure 5. Remove the value at r1, slide everything in (r1, r2] one step
    left, put the removed value at r2. Medium perturbation.

    Paper example (1-indexed r1=2, r2=5):
        1 2 3 4 5 6 7  ->  1 3 4 5 2 6 7
    """
    out = v.copy()
    t = out[r1]
    if r1 < r2:
        out[r1:r2] = out[r1 + 1:r2 + 1]
    else:                       # symmetric case: slide right
        out[r2 + 1:r1 + 1] = out[r2:r1]
    out[r2] = t
    return out


# ----------------------------------------------------------------------------
# Random position sampling with in-bounds guarantees
# ----------------------------------------------------------------------------
def rand_swap_positions(D: int, rng) -> tuple[int, int]:
    r1, r2 = rng.choice(D, size=2, replace=False)
    return int(r1), int(r2)


def rand_symmetry_positions(D: int, rng) -> tuple[int, int]:
    """
    r2 (block size) >= 1, and the span [r1-r2, r1+r2] must fit in [0, D-1].
    => r2 <= min(r1, D-1-r1). Sample the pivot r1 from [1, D-2] so that range
    is non-empty, then r2.
    """
    r1 = int(rng.integers(1, D - 1))                   # pivot in [1, D-2]
    max_blk = min(r1, D - 1 - r1)
    r2 = int(rng.integers(1, max_blk + 1))
    return r1, r2


def rand_shift_positions(D: int, rng) -> tuple[int, int]:
    r1, r2 = rng.choice(D, size=2, replace=False)
    return int(r1), int(r2)


# ----------------------------------------------------------------------------
# TESTS against the paper's own figures
# ----------------------------------------------------------------------------
def test_paper_figures():
    v = np.array([1, 2, 3, 4, 5, 6, 7])

    # Figure 3: swap(r1=3, r2=6) 1-indexed -> (2, 5) 0-indexed
    assert swap(v, 2, 5).tolist() == [1, 2, 6, 4, 5, 3, 7], swap(v, 2, 5).tolist()

    # Figure 4: symmetry(r1=4, r2=2) 1-indexed -> (3, 2) 0-indexed
    assert symmetry(v, 3, 2).tolist() == [1, 6, 5, 4, 3, 2, 7], symmetry(v, 3, 2).tolist()
    # explicit "two flanking blocks reversed then swapped" cross-check:
    assert symmetry(np.arange(1, 8), 3, 1).tolist() == [1, 2, 5, 4, 3, 6, 7]

    # Figure 5: shift(r1=2, r2=5) 1-indexed -> (1, 4) 0-indexed
    assert shift(v, 1, 4).tolist() == [1, 3, 4, 5, 2, 6, 7], shift(v, 1, 4).tolist()

    print("PASS  swap / symmetry / shift reproduce Figures 3, 4, 5 exactly")


def test_purity_and_bounds():
    rng = np.random.default_rng(1)
    for D in (7, 225, 2000):
        for _ in range(500):
            v = rng.uniform(-5, 5, D)
            snap = v.copy()
            a, b = rand_swap_positions(D, rng); o = swap(v, a, b)
            a, b = rand_symmetry_positions(D, rng); o = symmetry(v, a, b)
            a, b = rand_shift_positions(D, rng); o = shift(v, a, b)
            assert np.array_equal(v, snap), "operator mutated its input!"
            assert o.shape == v.shape
            # operators are permutations of the values -> multiset preserved,
            # so no clamping to [-5,5] is ever needed after a mutation.
            assert np.allclose(np.sort(o), np.sort(v))
    print("PASS  operators are pure, in-bounds, and value-multiset preserving "
          "(=> no clamp needed after mutation; clamp only after Eq.3/Eq.4)")


if __name__ == "__main__":
    test_paper_figures()
    test_purity_and_bounds()
