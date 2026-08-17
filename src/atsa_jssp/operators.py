"""
Mutation operators for ATSA — swap, symmetry, shift.

Ported from reference/operators_reference.py, which is VERIFIED against Sahman (2022)
Figures 3, 4, 5 (cross-checked against DTSA, Cinar et al. 2020, Figures 2-4).

Two paths, same contract, mirroring decoder.py:
  - the NumPy versions are the ORACLE (pure: they return a new array, never mutate input)
  - the `_nb` njit twins are what the ATSA kernel calls

Operators act on the CONTINUOUS vector (the ATSA design notes A1). They permute
values, so the value multiset is preserved and a mutated vector is still inside
[-5,5] => NEVER clamp after a mutation. Clamp only after Eq.3/Eq.4.
"""
from __future__ import annotations
import numpy as np
from atsa_jssp._compat import njit  # optional-Numba shim (pure-Python fallback)


# ----------------------------------------------------------------------------
# NumPy oracle
# ----------------------------------------------------------------------------
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
# Random position sampling with in-bounds guarantees (NumPy oracle)
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
# njit twins — what the ATSA kernel calls. Write into `out` to avoid allocating
# a fresh array per seed (this is the hot loop).
# ----------------------------------------------------------------------------
@njit(cache=True, nogil=True, inline="always")
def swap_nb(v, r1, r2, out):
    for k in range(v.shape[0]):
        out[k] = v[k]
    t = out[r1]
    out[r1] = out[r2]
    out[r2] = t


@njit(cache=True, nogil=True, inline="always")
def symmetry_nb(v, r1, r2, out):
    """Reverse the span [r1-r2, r1+r2] — 2*r2+1 elements centred on the pivot."""
    for k in range(v.shape[0]):
        out[k] = v[k]
    lo, hi = r1 - r2, r1 + r2
    while lo < hi:
        t = out[lo]
        out[lo] = out[hi]
        out[hi] = t
        lo += 1
        hi -= 1


@njit(cache=True, nogil=True, inline="always")
def shift_nb(v, r1, r2, out):
    for k in range(v.shape[0]):
        out[k] = v[k]
    t = out[r1]
    if r1 < r2:
        for k in range(r1, r2):
            out[k] = out[k + 1]
    else:
        for k in range(r1, r2, -1):
            out[k] = out[k - 1]
    out[r2] = t


@njit(cache=True, nogil=True, inline="always")
def rand_two_positions_nb(D):
    """Two distinct positions in [0, D-1]. Equivalent to choice(D, 2, replace=False)."""
    r1 = np.random.randint(0, D)
    r2 = np.random.randint(0, D - 1)
    if r2 >= r1:
        r2 += 1
    return r1, r2


@njit(cache=True, nogil=True, inline="always")
def rand_symmetry_positions_nb(D):
    r1 = np.random.randint(1, D - 1)                   # pivot in [1, D-2]
    max_blk = min(r1, D - 1 - r1)
    r2 = np.random.randint(1, max_blk + 1)
    return r1, r2
