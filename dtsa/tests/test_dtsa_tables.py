"""
Checksums on the DTSA paper's tables, as parsed by an offline table-extraction step from the source paper.

WHY THIS FILE EXISTS. On the ATSA side of this project a hand-transcribed reference column was
wrong on 30 of 40 rows and fabricated an entire "size-dependent drift" finding
(the project design notes §8 D1, §9 items 14-17). What caught it was Sahman's Table 5 printing its own AVG
row -- an independent checksum that was available the whole time and that nobody ran.

**The DTSA paper has no AVG row.** That mechanism does not exist here. The substitute is a
derived-column identity: most of its result tables print a mean AND that mean's error against a
stated optimum, so the error column can be recomputed from the mean column. Every row that has
both is asserted below. Rows that do not are named explicitly and marked unusable as gate targets.

Run:  uv run pytest dtsa/tests -q
"""
from __future__ import annotations

import pytest
from dtsa_tables import OPTIMA, TABLE1, TABLE4, TABLE5, TABLE15

# Printed to 2 dp, so a faithful recomputation can differ by up to half a unit in the last place.
TOL = 0.006


# --------------------------------------------------------------------------------------------
# Table 5 -- the conventional identity, and the one row where the paper contradicts itself
# --------------------------------------------------------------------------------------------
def _t5_rows():
    return [(p, a) for p, d in TABLE5.items() for a in d["algorithms"]]


@pytest.mark.parametrize("problem,algo", _t5_rows())
def test_table5_mean_error_identity(problem, algo):
    """Mean Error == (Mean - Optimum)/Optimum * 100, against the table's own Optimum column.

    KROE100 is excluded and tested separately below: the paper computed that block's error
    column with the WRONG optimum. That is a defect in the paper, not in this parse, and it is
    asserted explicitly rather than tolerated silently.
    """
    if problem == "KROE100":
        pytest.skip("see test_table5_kroe100_uses_the_wrong_optimum")
    d = TABLE5[problem]
    opt, s = d["optimum"], d["algorithms"][algo]
    assert (s["mean"] - opt) / opt * 100 == pytest.approx(s["mean_error"], abs=TOL)


def test_table5_kroe100_uses_the_wrong_optimum():
    """
    Table 5 (journal p. 886) prints KROE100's optimum as 22068 -- which is correct, it is
    TSPLIB's value for kroE100. But every one of its five Mean Error entries was computed with
    ~22141 instead, and 22141 is **KROB100's** optimum, printed three rows higher in the same
    table. Found by the checksum, not by eye.

    The MEAN column survives this. Five independent rows agreeing on the same implied optimum to
    within rounding cannot happen by accident, so the means are mutually consistent and remain
    usable; only the error column and the pairing with the printed optimum are wrong.
    """
    d = TABLE5["KROE100"]
    assert d["optimum"] == 22068, "the printed Optimum cell should still read 22068"
    assert TABLE5["KROB100"]["optimum"] == 22141

    implied = [s["mean"] / (1 + s["mean_error"] / 100) for s in d["algorithms"].values()]
    assert len(implied) == 5
    # All five agree with each other ...
    assert max(implied) - min(implied) < 2.0
    # ... on KROB100's optimum, not their own.
    assert sum(implied) / len(implied) == pytest.approx(22141, abs=1.0)
    for v in implied:
        assert abs(v - 22068) > 50, "if this fails the defect is gone and the skip above is stale"


# --------------------------------------------------------------------------------------------
# Table 1 -- the non-standard denominator (U8), confirmed by test rather than by spot-check
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("row", sorted(TABLE1))
def test_table1_re_divides_by_the_result_not_the_optimum(row):
    """
    RE(%) == (TSA+2Opt Mean - Optimum) / (TSA+2Opt Mean) * 100.

    Dividing by the IMPROVED value rather than the reference inflates every figure. It is the
    same defect documented for the ATSA paper in the project design notes §8 R2, in the same research
    lineage, and it contradicts this paper's own Eq. 5 (p. 883) as well as its Tables 2, 3, 5
    and 15. Holding across all nine rows is what makes this identity Table 1's checksum.
    """
    opt = OPTIMA["BERLIN52"][0]          # 7542, rounded EUC_2D -- see test_berlin52.py
    s = TABLE1[row]
    r = s["tsa_2opt_mean"]
    assert (r - opt) / r * 100 == pytest.approx(s["re"], abs=TOL)


def test_table1_conventional_denominator_would_be_wrong():
    """The negative control: the conventional formula must NOT reproduce Table 1."""
    opt = OPTIMA["BERLIN52"][0]
    disagreeing = [
        row for row, s in TABLE1.items()
        if abs((s["tsa_2opt_mean"] - opt) / opt * 100 - s["re"]) >= TOL
    ]
    # The three rows sitting exactly at the optimum are 0.00 either way and cannot discriminate.
    assert len(disagreeing) == 6


def test_table1_swap_rows_are_duplicated():
    """
    U16. `swap(current tree)` and `swap(best tree)` are IDENTICAL in all four columns:
    8133.00 / 7863.00 / 0.00 / 4.08.

    Seeding from the current tree and seeding from the global best are different algorithms.
    Over 30 stochastic runs they cannot land on the same mean to 2 dp in two separate columns
    with identical zero variance. The likely explanation is a row duplicated at typesetting.

    CONSEQUENCE: `swap(best tree)` carries no independent published value, so it cannot be a
    Gate 1 validation target and is EXCLUDED there (the DTSA adaptation notes D5).

    This test pins the observation so that a future change to the extractor cannot quietly make
    the duplication disappear -- if these rows ever differ, either the paper was re-read wrongly
    or the parse drifted, and both need a human.
    """
    a, b = TABLE1["swap(current tree)"], TABLE1["swap(best tree)"]
    assert a == b, "the duplication is the finding; if it is gone, re-read the page"
    assert a == {"tsa_mean": 8133.0, "tsa_2opt_mean": 7863.0, "std": 0.0, "re": 4.08}

    # ... and it is the ONLY collision in the table, which is what makes duplication the likely
    # explanation rather than a table-wide artefact.
    collisions = [
        (x, y) for i, x in enumerate(sorted(TABLE1)) for y in sorted(TABLE1)[i + 1:]
        if TABLE1[x] == TABLE1[y]
    ]
    assert collisions == [("swap(best tree)", "swap(current tree)")]


def test_table1_std_column_describes_the_post_2opt_figures():
    """
    the DTSA specification §5.6. Table 1 has one Std.Dev. column and two mean columns. It belongs to the
    post-2-opt one, and `symmetry(current tree)` proves it:

      Table 1 is on rounded EUC_2D (U9), so every tour length is an integer and a 30-run mean is
      a multiple of 1/30. Its pre-2-opt mean is 7683.73 -- non-integer, so the runs differed.
      A Std.Dev. of 0.00 is therefore impossible for that column.

    Why it matters: the pre-2-opt column is `DTSA-core`'s ONLY published target, and the paper
    gives it no spread at all. Every tolerance we apply to it is ours, not the paper's.
    """
    s = TABLE1["symmetry(current tree)"]
    assert s["std"] == 0.0
    assert s["tsa_2opt_mean"] == float(int(s["tsa_2opt_mean"]))    # integer: all runs agreed
    assert s["tsa_mean"] != float(int(s["tsa_mean"]))              # non-integer: runs differed
    assert abs(s["tsa_mean"] * 30 - round(s["tsa_mean"] * 30)) < 0.5, \
        "a 30-run mean of integers must be a multiple of 1/30"

    zero_std = sorted(r for r, v in TABLE1.items() if v["std"] == 0.0)
    assert zero_std == ["swap(best tree)", "swap(current tree)",
                        "symmetry(current tree)", "symmetry(random tree)"]
    assert len(TABLE1) - len(zero_std) == 5, "only 5 rows carry a non-zero published spread"


def test_table1_publishes_a_pre_2opt_column():
    """
    Table 1 is the ONLY place in the paper with a no-local-search DTSA number, which is why
    the DTSA adaptation notes D5 makes it the primary gate: it is the only published target for
    the `DTSA-core` configuration we actually ship. Guard that the column exists and is
    genuinely distinct from the post-2-opt one.
    """
    assert len(TABLE1) == 9
    for row, s in TABLE1.items():
        assert s["tsa_mean"] >= s["tsa_2opt_mean"], f"{row}: 2-opt made it worse?"
    improved = [r for r, s in TABLE1.items() if s["tsa_mean"] > s["tsa_2opt_mean"]]
    assert len(improved) == 9, "2-opt should improve every configuration"


# --------------------------------------------------------------------------------------------
# Table 15 -- same identity, but the distance convention varies per instance (U9)
# --------------------------------------------------------------------------------------------
def _t15_rows():
    return [(p, m) for p, d in TABLE15.items() for m in d]


@pytest.mark.parametrize("problem,method", _t15_rows())
def test_table15_re_identity(problem, method):
    """RE == (Mean - Optimum)/Optimum * 100 under ONE of the instance's published optima."""
    if problem not in OPTIMA:
        pytest.skip(f"{problem} has no optimum published in §5 -- unverifiable")
    s = TABLE15[problem][method]
    cands = [o for o in OPTIMA[problem] if o is not None]
    assert any(
        (s["mean"] - o) / o * 100 == pytest.approx(s["re"], abs=TOL) for o in cands
    ), f"{problem}/{method}: RE {s['re']} matches none of {cands}"


def test_table15_unverifiable_rows_are_exactly_the_two_known_ones():
    """EIL76 and CH150 appear in Table 15 but not in §5's optimum list. Pin the count so it
    cannot silently grow if the parser drifts."""
    missing = sorted(p for p in TABLE15 if p not in OPTIMA)
    assert missing == ["CH150", "EIL76"]
    assert sum(len(TABLE15[p]) for p in missing) == 8


def test_table15_berlin52_and_kroa100_use_the_unrounded_optimum():
    """
    The paper switches distance convention between tables without saying so (U9). Table 15 uses
    the UNROUNDED optimum for the two instances that publish both, while Tables 1, 4 and 5 use
    the rounded one. Asserted so that a gate is never run under the wrong convention.
    """
    for problem in ("BERLIN52", "KROA100"):
        rounded, unrounded = OPTIMA[problem]
        assert unrounded is not None
        s = TABLE15[problem]["DTSA"]
        assert (s["mean"] - unrounded) / unrounded * 100 == pytest.approx(s["re"], abs=TOL)
        assert (s["mean"] - rounded) / rounded * 100 != pytest.approx(s["re"], abs=TOL)


# --------------------------------------------------------------------------------------------
# Table 4 -- NO internal checksum. Two witnesses only.
# --------------------------------------------------------------------------------------------
# Witness 2: an independent VISUAL read of the rasterised table, rendered full-width at 450 dpi
# to an offline reading of the source paper and read cell by cell. Typed from that image, NOT copied
# from the extractor's output.
TABLE4_VISUAL_WITNESS = {
    "best":  {"SA": 8186.40, "ACO": 8240.40, "STA": 7544.40, "DTSA": 7542.00},
    "mean":  {"SA": 8983.80, "ACO": 8777.60, "STA": 8247.20, "DTSA": 7689.17},
    "worse": {"SA": 9585.80, "ACO": 9151.30, "STA": 8630.50, "DTSA": 7929.00},
    "std":   {"SA": 380.10,  "ACO": 267.11,  "STA": 273.45,  "DTSA": 108.40},
}


@pytest.mark.parametrize("stat", ["best", "mean", "worse", "std"])
@pytest.mark.parametrize("algo", ["SA", "ACO", "STA", "DTSA"])
def test_table4_two_witness(algo, stat):
    """
    TWO-WITNESS CHECK, AND IT IS WEAKER THAN A DERIVED IDENTITY. READ THIS BEFORE TRUSTING
    TABLE 4.

    Table 4 (journal p. 885) prints Best / Mean / Worse / Std.Dev. and NO error column, so
    there is nothing inside it to recompute -- no identity can be asserted the way it is for
    Tables 1, 5 and 15. The best available substitute is two independent extraction routes:

        witness 1 : the PDF text layer, parsed by an offline table-extraction step from the source paper
        witness 2 : a visual read of the page rasterised at 450 dpi, typed above by hand

    Agreement rules out a text-layer extraction fault and a transcription slip. It does NOT
    rule out a shared upstream error, and it does not validate the numbers against anything
    the paper computed. A derived identity is self-checking; this is two people agreeing.

    Consequence, recorded in the DTSA adaptation notes D5: Table 4 was DEMOTED from primary gate
    to secondary, and Table 1 -- which does have an identity, and which publishes the
    pre-2-opt column we actually need -- was promoted in its place.
    """
    assert TABLE4["BERLIN52"][algo][stat] == pytest.approx(
        TABLE4_VISUAL_WITNESS[stat][algo], abs=0.005
    )


def test_table4_internal_ordering_sanity():
    """Best <= Mean <= Worse for every algorithm. Weak, but it is free and it would catch a
    column swap, which is the single most likely two-witness-surviving failure."""
    for algo, s in TABLE4["BERLIN52"].items():
        assert s["best"] <= s["mean"] <= s["worse"], algo


def test_table4_dtsa_best_is_the_rounded_optimum():
    """
    DTSA's Best of 7542.00 IS berlin52's rounded-EUC_2D optimum, which is independent evidence
    that Table 4 is on the rounded convention (U9) -- and the only external cross-check this
    table has.
    """
    assert TABLE4["BERLIN52"]["DTSA"]["best"] == pytest.approx(OPTIMA["BERLIN52"][0], abs=0.005)
