"""
TSP objective for the DTSA validation gates.

Cinar et al. Eq. 4 (p. 883):  f = d(n,1) + sum_{c=1..n-1} d(c, c+1)   -- a closed tour.

Distance convention is a FLAG because the paper switches between the two without saying so
(U9, settled in the DTSA specification §5.4). The default is rounded TSPLIB EUC_2D, which is what
Tables 1, 4 and 5 use -- i.e. what Gates 1, 2 and 3 need. Tables 8 and 15 are unrounded.
"""
from __future__ import annotations

import pathlib

import numpy as np

from tsplib_io import nint, read_tsp


class TSP:
    """A symmetric Euclidean TSP with a precomputed distance matrix."""

    def __init__(self, name: str, coords: list[tuple[float, float]], *, rounded: bool = True):
        self.name = name
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n = len(coords)
        self.rounded = rounded

        dx = self.coords[:, 0][:, None] - self.coords[None, :, 0]
        dy = self.coords[:, 1][:, None] - self.coords[None, :, 1]
        d = np.sqrt(dx * dx + dy * dy)
        if rounded:
            # TSPLIB nint = int(x + 0.5). Vectorised form of tsplib_io.nint; the equivalence is
            # asserted in tests/test_tsp.py.
            d = np.floor(d + 0.5)
        self.dist = d

    @property
    def D(self) -> int:
        """Problem dimension -- the number of decision variables. For TSP, the city count."""
        return self.n

    def tour_length(self, tour: np.ndarray) -> float:
        """Eq. 4. `tour` is a permutation of 0..n-1."""
        return float(self.dist[tour, np.roll(tour, -1)].sum())

    def __repr__(self) -> str:
        conv = "rounded EUC_2D" if self.rounded else "unrounded Euclidean"
        return f"TSP({self.name}, n={self.n}, {conv})"


def load_tsp(path: str | pathlib.Path, *, rounded: bool = True) -> TSP:
    name, coords, ewt = read_tsp(pathlib.Path(path))
    if ewt and ewt != "EUC_2D":
        raise ValueError(
            f"{name}: EDGE_WEIGHT_TYPE is {ewt!r}, not EUC_2D. Cinar et al. §5 (p. 883) "
            "explicitly exclude GEO instances evaluated as Euclidean -- so do we."
        )
    return TSP(name, coords, rounded=rounded)


def nearest_neighbour_tour(problem: TSP, start: int = 0) -> np.ndarray:
    """
    Fig. 6 line 5: "Determine the first tree as nearest neighbor tour."

    U12: the paper does not say which city the tour starts from. `start` defaults to city 0
    (city 1 in TSPLIB's 1-indexed numbering), which is the obvious literal reading. Ties are
    broken by lowest index so the tour is deterministic.
    """
    n = problem.n
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        cur = min(unvisited, key=lambda j: (problem.dist[cur, j], j))
        unvisited.remove(cur)
        tour.append(cur)
    return np.array(tour, dtype=np.int64)


def best_nearest_neighbour_tour(problem: TSP) -> tuple[np.ndarray, int]:
    """
    The shortest NN tour over all possible start cities, and the city that produced it.

    ⚠️ NOT A DEFAULT AND NOT A MORE FAITHFUL READING. Fig. 6 line 5 says "nearest neighbor
    tour", singular. Taking the best of n starts is a better-performing choice, not a more
    literal one, so it exists only as the U12 arm of the Part B diagnostic
    (the project log D004). Adopting it would be tuning.
    """
    best_tour, best_len, best_start = None, float("inf"), -1
    for s in range(problem.n):
        t = nearest_neighbour_tour(problem, start=s)
        L = problem.tour_length(t)
        if L < best_len:
            best_tour, best_len, best_start = t, L, s
    return best_tour, best_start


def check_nint_matches(problem: TSP) -> bool:
    """The vectorised floor(d+0.5) must agree with tsplib_io.nint on every pair."""
    c = problem.coords
    for i in range(problem.n):
        for j in range(problem.n):
            d = float(np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1]))
            if problem.dist[i, j] != nint(d):
                return False
    return True
