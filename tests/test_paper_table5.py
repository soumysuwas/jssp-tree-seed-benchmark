"""
Checksum tests for our transcription of the paper's Table 5.

WHY THIS FILE EXISTS. The ATSA Mean/Min column was hand-transcribed into a dict and was wrong
from ta21 onward — systematically low, with the error growing with D (ta21 off by 21, ta51 by 83,
ta71 by 237). Because the error grew with problem size, comparing our runs against it produced a
clean, statistically overwhelming "our reproduction degrades with problem size" effect
(z = +9.1 at D=2000, corr(D, diff%) = +0.72). It was entirely an artefact of the transcription.
The real effect is corr = -0.02: no drift at all.

Table 5 prints its OWN AVG row. That is an independent checksum on every column, it was
available the whole time, and running it takes one line. These tests run it on every column of
every algorithm, so a bad transcription can never again be mistaken for a discovery.

Rule: never hand-edit src/atsa_jssp/paper_table5.py. It was produced once, offline, from the
source paper's Table 5, and is the only copy of Şahman (2022) Table 5 in this repository. (The
DTSA paper's tables, Cinar 2020, live separately in dtsa/dtsa_tables.py.)
"""
from __future__ import annotations

import pytest

from atsa_jssp.instance import PAPER_INSTANCES          # single source of the 40-instance scope
from atsa_jssp.paper_table5 import ALGOS, AVG, STATS, TABLE5


def test_table5_has_all_40_paper_instances():
    assert len(PAPER_INSTANCES) == 40
    assert sorted(TABLE5) == sorted(PAPER_INSTANCES)
    assert len(TABLE5) == 40


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("stat", STATS)
def test_column_matches_paper_avg_row(algo, stat):
    """
    THE CHECKSUM. The mean of each transcribed column must equal the AVG row the paper prints
    for it. The AVG row is stated to 2dp over 40 values, so rounding permits ~0.05 in principle.

    HOW TIGHT THIS ACTUALLY IS (measured). As transcribed, 19 of the 20 columns reconcile to the
    printed AVG within 0.005. The one exception is ATSA/mean: the column averages 2839.3675 vs the
    printed 2839.35, a residual of +0.0175 (its column sum is ~0.7 higher than the printed AVG
    implies). We do NOT know whether that is the paper's own AVG rounding or a single cell differing
    by ~0.7; it cannot be resolved without the source PDF, which is not redistributed here. So the
    tolerance below is set to 0.02 — the tightest bound that still passes every column with the
    values exactly as transcribed. Read it as "±0.02", not as an exact match.

    If this fails, the transcription is wrong — not the paper. Do not widen the tolerance beyond 0.02.
    """
    col = [TABLE5[i][algo][stat] for i in PAPER_INSTANCES]
    got = sum(col) / len(col)
    want = AVG[algo][stat]
    assert got == pytest.approx(want, abs=0.02), (
        f"{algo}.{stat}: column averages {got:.4f}, paper's AVG row says {want:.2f} "
        f"(delta {got - want:+.4f}). The transcription is wrong."
    )


def test_atsa_mean_column_checksum_explicitly():
    """
    The specific column that was wrong, called out by name so a regression is unmissable.
    As transcribed it averages 2839.3675 vs the printed 2839.35 (residual +0.0175); see the
    parametrised checksum's docstring for why that residual is disclosed, not smoothed over.
    """
    col = [TABLE5[i]["ATSA"]["mean"] for i in PAPER_INSTANCES]
    got = sum(col) / len(col)
    assert got == pytest.approx(2839.35, abs=0.02), (
        f"ATSA mean column averages {got:.4f}; the paper's Table 5 AVG row says 2839.35 "
        f"(residual {got - 2839.35:+.4f}). A previous transcription averaged 2793.38 (off by 46) "
        f"and fabricated a size-dependent drift finding."
    )


def test_atsa_beats_every_baseline_on_every_instance():
    """
    The paper's W/T row claims ATSA wins 48/48. Whatever '48' means over 40 instances (defect
    P3), the transcribed table must at least be internally consistent with ATSA winning
    everywhere — a sanity check that columns were not swapped during transcription.
    """
    for inst in PAPER_INSTANCES:
        atsa = TABLE5[inst]["ATSA"]["mean"]
        for algo in ("PSO", "TSA", "ABC", "GWO"):
            assert atsa < TABLE5[inst][algo]["mean"], f"{inst}: ATSA {atsa} !< {algo}"


def test_known_spot_values():
    """Anchors read straight off the PDF, including the row where the old transcription broke."""
    assert TABLE5["ta01"]["ATSA"]["mean"] == 1444.8
    assert TABLE5["ta01"]["ATSA"]["min"] == 1347
    assert TABLE5["ta01"]["TSA"]["mean"] == 1724.55
    # ta21 is where the hand-transcription first went wrong: it had 2012.1, the paper says 1990.6
    assert TABLE5["ta21"]["ATSA"]["mean"] == 1990.6
    # ta71 was off by 237: it had 6204.4, the paper says 6441
    assert TABLE5["ta71"]["ATSA"]["mean"] == 6441
    assert TABLE5["ta71"]["TSA"]["mean"] == 8371.05
