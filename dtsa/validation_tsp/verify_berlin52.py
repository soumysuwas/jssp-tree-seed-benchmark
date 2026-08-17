#!/usr/bin/env python3
"""
Settle U9 empirically: which distance convention does each table of Cinar et al. (2020) use?

The paper lists BERLIN52's optimum as "7542/7544.37" and KROA100's as "21282/21285.44" (§5,
journal p. 883), then quotes one or the other in different tables without ever saying which
convention it is using. Guessing wrong puts a small, systematic offset between our numbers and
the paper's -- which is precisely the failure mode that manufactured a false finding on the ATSA
side of this project (the project design notes §8 D1).

So we do not argue about it. We take TSPLIB's own published optimal tour, recompute its length
from the coordinates under both conventions, and see which number falls out.

Usage:  uv run python dtsa/verify_berlin52.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tsplib_io import read_opt_tour, read_tsp, tour_length  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "data" / "tsplib"

# The two published values per instance, as printed in Cinar et al. (2020) §5, p. 883.
# These are the ONLY hand-typed numbers in this file, and the whole point of the script is to
# check them against a recomputation rather than trust them.
PUBLISHED = {
    "berlin52": (7542.0, 7544.37),
    "kroA100": (21282.0, 21285.44),
}


def check(stem: str) -> dict:
    _, coords, ewt = read_tsp(DATA / f"{stem}.tsp")
    tour = read_opt_tour(DATA / f"{stem}.opt.tour")
    if sorted(tour) != list(range(len(coords))):
        raise SystemExit(f"FATAL: {stem}.opt.tour is not a permutation of the {len(coords)} cities")

    rounded = tour_length(tour, coords, rounded=True)
    unrounded = tour_length(tour, coords, rounded=False)
    pub_int, pub_frac = PUBLISHED[stem]

    ok_r = abs(rounded - pub_int) < 0.5
    ok_u = abs(unrounded - pub_frac) < 0.005

    print(f"\n{stem}  (EDGE_WEIGHT_TYPE = {ewt}, n = {len(coords)})")
    print(f"  rounded EUC_2D, nint per edge : {rounded:12.4f}   "
          f"vs published {pub_int:10.2f}   {'MATCH' if ok_r else 'NO MATCH'}")
    print(f"  unrounded Euclidean           : {unrounded:12.4f}   "
          f"vs published {pub_frac:10.2f}   {'MATCH' if ok_u else 'NO MATCH'}")
    return {"stem": stem, "rounded": rounded, "unrounded": unrounded, "ok_r": ok_r, "ok_u": ok_u}


def main() -> None:
    print("U9 — which distance convention does each table use?")
    print("=" * 72)
    results = [check(s) for s in PUBLISHED if (DATA / f"{s}.opt.tour").exists()]
    if not results:
        raise SystemExit("FATAL: no .opt.tour files present — run dtsa/fetch_tsplib.py first")

    bad = [r for r in results if not (r["ok_r"] and r["ok_u"])]
    if bad:
        raise SystemExit(
            "FATAL: neither convention reproduces the published optimum for "
            f"{[r['stem'] for r in bad]}. The data file is wrong, or the parser is. STOP."
        )

    print("\n" + "=" * 72)
    print("RESOLVED. Both published values are real; they are the SAME tour measured two ways.")
    print("  integer-looking optimum (7542, 21282)  <- TSPLIB EUC_2D, nint() per edge")
    print("  fractional optimum (7544.37, 21285.44) <- unrounded Euclidean")
    print("\nTherefore, by which optimum each table quotes:")
    print("  Table 1  (p. 885, BERLIN52, best 7542.00)   -> ROUNDED EUC_2D")
    print("  Table 4  (p. 885, BERLIN52, best 7542.00)   -> ROUNDED EUC_2D")
    print("  Table 5  (p. 886, KROA100, optimum 21282)   -> ROUNDED EUC_2D")
    print("  Table 8  (p. 887, BERLIN52, DTSA 7544.37)   -> UNROUNDED Euclidean")
    print("\nGates 1, 2 and 3 all run under ROUNDED EUC_2D. Experiment 5 (Tables 6-14) does not.")


if __name__ == "__main__":
    main()
