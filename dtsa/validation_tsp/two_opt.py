"""
2-opt local search -- Fig. 6 line 38, "Apply the 2-opt algorithm using the best individual."

That single sentence is the whole of the paper's specification (U6). Everything below beyond
"reverse a segment to remove a crossing" is OUR CHOICE and is flagged as such:

  variant="first"  (DEFAULT, ours)  accept the first improving move found
  variant="best"                    scan all moves, take the best, repeat

Neither is more literal than the other, because the paper says nothing. First-improvement is the
default because it is the cheaper and more common reading of "2-opt" and because it bounds the
uncounted budget better -- but it IS a choice, and Gate 2's outcome must be read with that in
mind (the DTSA adaptation notes D5).

FE ACCOUNTING -- READ THIS. Fig. 6 never increments `fes` for 2-opt: line 31 is the only
increment and it sits inside the while loop, while line 38 is after `end while`. So DTSA's
published numbers include an uncounted local-search budget of unstated size (U7). We therefore
count 2-opt's evaluations and report them SEPARATELY -- never folded into `fes`
(the DTSA adaptation notes D3).

The unit is "candidate solutions evaluated". A 2-opt move is scored by an O(1) delta rather than
an O(n) full retraversal, so this count overstates 2-opt's WALL-CLOCK cost relative to a DTSA
seed evaluation -- but candidate-solutions-evaluated is the correct unit for an FE budget, which
is what the fairness question is about.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TwoOptResult:
    tour: np.ndarray
    length: float
    evaluations: int          # candidate solutions evaluated -- NEVER added to `fes`
    improving_moves: int
    passes: int


def two_opt(tour: np.ndarray, dist: np.ndarray, *, variant: str = "first",
            max_passes: int = 1000) -> TwoOptResult:
    """
    Standard 2-opt on a closed tour: reverse tour[i..j] and keep it if the tour gets shorter.

    The move breaks edges (i-1, i) and (j, j+1) and rebuilds them as (i-1, j) and (i, j+1); every
    edge inside the reversed segment survives, because reversing an undirected path preserves its
    edge set. Hence the O(1) delta.
    """
    if variant not in ("first", "best"):
        raise ValueError(f"two_opt: unknown variant {variant!r}")

    t = tour.copy()
    n = len(t)
    evaluations = 0
    improving = 0
    passes = 0

    improved = True
    while improved and passes < max_passes:
        improved = False
        passes += 1
        best_delta = 0.0
        best_ij = None

        for i in range(1, n - 1):
            a, b = t[i - 1], t[i]
            for j in range(i + 1, n):
                c = t[j]
                d = t[(j + 1) % n]
                if d == a:                       # the move would rebuild the same tour
                    continue
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                evaluations += 1
                if delta < -1e-12:
                    if variant == "first":
                        t[i:j + 1] = t[i:j + 1][::-1]
                        improving += 1
                        improved = True
                        break
                    if delta < best_delta:
                        best_delta, best_ij = delta, (i, j)
            if improved and variant == "first":
                break

        if variant == "best" and best_ij is not None:
            i, j = best_ij
            t[i:j + 1] = t[i:j + 1][::-1]
            improving += 1
            improved = True

    length = float(dist[t, np.roll(t, -1)].sum())
    return TwoOptResult(tour=t, length=length, evaluations=evaluations,
                        improving_moves=improving, passes=passes)
