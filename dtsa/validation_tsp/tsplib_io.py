"""
Minimal TSPLIB readers + the two distance conventions. NOT algorithm code — parsing and a
tour-length function, nothing that searches.

The two conventions matter because Cinar et al. (2020) switch between them without saying so
(`the DTSA specification` U9):

  EUC_2D (TSPLIB official):  d_ij = nint(sqrt(dx^2 + dy^2))   -- each EDGE rounded to an integer
  unrounded Euclidean:       d_ij =      sqrt(dx^2 + dy^2)

`nint` in TSPLIB is defined as `int(x + 0.5)`, i.e. round-half-up, NOT Python's banker's rounding.
Using `round()` here would be a silent off-by-one on exact .5 edges.
"""
from __future__ import annotations

import math
import pathlib


def read_tsp(path: pathlib.Path) -> tuple[str, list[tuple[float, float]], str]:
    """Return (name, coords 0-indexed, edge_weight_type) from a TSPLIB .tsp file."""
    name, ewt = path.stem, ""
    coords: list[tuple[float, float]] = []
    in_coords = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("NAME"):
            name = line.split(":", 1)[1].strip() if ":" in line else name
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            ewt = line.split(":", 1)[1].strip() if ":" in line else ""
        elif line.startswith("NODE_COORD_SECTION"):
            in_coords = True
        elif line in ("EOF", "DISPLAY_DATA_SECTION"):
            in_coords = False
        elif in_coords:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
    if not coords:
        raise ValueError(f"{path}: no NODE_COORD_SECTION parsed")
    return name, coords, ewt


def read_opt_tour(path: pathlib.Path) -> list[int]:
    """Return the tour as 0-indexed city numbers from a TSPLIB .opt.tour file."""
    tour: list[int] = []
    in_tour = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("TOUR_SECTION"):
            in_tour = True
        elif line == "EOF":
            in_tour = False
        elif in_tour:
            for tok in line.split():
                v = int(tok)
                if v == -1:
                    in_tour = False
                    break
                tour.append(v - 1)
    if not tour:
        raise ValueError(f"{path}: no TOUR_SECTION parsed")
    return tour


def nint(x: float) -> int:
    """TSPLIB's nint: int(x + 0.5). Deliberately not Python's round() (banker's rounding)."""
    return int(x + 0.5)


def tour_length(tour: list[int], coords: list[tuple[float, float]], *, rounded: bool) -> float:
    """
    Closed-tour length, matching Cinar et al. Eq. 4 (p. 883):  f = d(n,1) + sum d(c, c+1).

    rounded=True  -> TSPLIB EUC_2D, each edge nint()-rounded (integer total)
    rounded=False -> unrounded Euclidean
    """
    total = 0.0
    n = len(tour)
    for i in range(n):
        (x1, y1) = coords[tour[i]]
        (x2, y2) = coords[tour[(i + 1) % n]]
        d = math.hypot(x1 - x2, y1 - y2)
        total += nint(d) if rounded else d
    return total
