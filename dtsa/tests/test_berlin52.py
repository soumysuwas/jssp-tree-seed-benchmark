"""
U9 -- the distance-rounding ambiguity, settled empirically and then frozen as a test.

Cinar et al. (2020) §5, p. 883 list BERLIN52's optimum as "7542/7544.37" and KROA100's as
"21282/21285.44", then quote one or the other in different tables without ever saying which
convention is in force. Running a validation gate under the wrong one puts a small, systematic
offset between our numbers and the paper's -- exactly the failure mode that manufactured a false
finding on the ATSA side (the project design notes §8 D1).

These tests recompute the length of TSPLIB's own published optimal tour from the fetched
coordinates. They double as a data-integrity check on data/tsplib/: a mirror with corrupted
coordinates cannot reproduce both published values.
"""
from __future__ import annotations

import pathlib

import pytest
from dtsa_tables import OPTIMA, TABLE1, TABLE4, TABLE5
from tsplib_io import nint, read_opt_tour, read_tsp, tour_length

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "tsplib"

pytestmark = pytest.mark.skipif(
    not (DATA / "berlin52.opt.tour").exists(),
    reason="run `uv run python dtsa/fetch_tsplib.py` first",
)


def _length(stem: str, *, rounded: bool) -> float:
    _, coords, _ = read_tsp(DATA / f"{stem}.tsp")
    tour = read_opt_tour(DATA / f"{stem}.opt.tour")
    assert sorted(tour) == list(range(len(coords))), f"{stem}.opt.tour is not a permutation"
    return tour_length(tour, coords, rounded=rounded)


def test_nint_is_round_half_up_not_bankers():
    """TSPLIB defines nint(x) = int(x + 0.5). Python's round() does banker's rounding and would
    silently differ on exact .5 edges."""
    assert nint(0.5) == 1 and round(0.5) == 0
    assert nint(1.5) == 2 and round(1.5) == 2
    assert nint(2.5) == 3 and round(2.5) == 2


@pytest.mark.parametrize("stem,rounded_opt,unrounded_opt", [
    ("berlin52", 7542.0, 7544.37),
    ("kroA100", 21282.0, 21285.44),
])
def test_both_published_optima_are_the_same_tour_measured_two_ways(stem, rounded_opt,
                                                                  unrounded_opt):
    assert _length(stem, rounded=True) == pytest.approx(rounded_opt, abs=0.5)
    assert _length(stem, rounded=False) == pytest.approx(unrounded_opt, abs=0.005)


def test_berlin52_exact_values():
    """Pin the exact numbers so a data swap is loud rather than subtle."""
    assert _length("berlin52", rounded=True) == pytest.approx(7542.0000, abs=1e-6)
    assert _length("berlin52", rounded=False) == pytest.approx(7544.3659, abs=1e-3)


def test_paper_optima_list_matches_the_recomputation():
    """§5's parenthesised pairs, as parsed into OPTIMA, are exactly what the tour recomputes."""
    for stem, name in [("berlin52", "BERLIN52"), ("kroA100", "KROA100")]:
        rounded, unrounded = OPTIMA[name]
        assert unrounded is not None
        assert _length(stem, rounded=True) == pytest.approx(rounded, abs=0.5)
        assert _length(stem, rounded=False) == pytest.approx(unrounded, abs=0.005)


def test_gate_tables_are_all_on_the_rounded_convention():
    """
    U9 RESOLVED, per table. Gates 1, 2 and 3 (the DTSA adaptation notes D5) all run under rounded
    EUC_2D; Experiment 5 and Table 15 do not. Asserted here so a gate can never be run under the
    wrong one.
    """
    rounded = OPTIMA["BERLIN52"][0]
    # Gate 1, Table 1: three configurations reach exactly the rounded optimum.
    at_opt = [r for r, s in TABLE1.items() if s["tsa_2opt_mean"] == pytest.approx(rounded, abs=.5)]
    assert len(at_opt) == 2, at_opt
    # Gate 2, Table 4: DTSA's Best is the rounded optimum.
    assert TABLE4["BERLIN52"]["DTSA"]["best"] == pytest.approx(rounded, abs=0.005)
    # Gate 3, Table 5: its own Optimum column is the rounded value.
    assert TABLE5["KROA100"]["optimum"] == pytest.approx(OPTIMA["KROA100"][0], abs=0.5)
