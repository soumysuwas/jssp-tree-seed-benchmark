#!/usr/bin/env python3
"""
Compute the headline ATSA-reproduction numbers used in the README, directly from the
committed CSVs and the committed paper table. Nothing here is hand-typed.

Scope: the 40 paper instances listed in data/paper_scope_40.txt (identical to
PAPER_INSTANCES). Every aggregate printed here is over exactly those 40 instances.

    python scripts/compute_headline.py

Reads:
    results/atsa/atsa_ta01.csv, results/atsa/atsa_ta02_ta80.csv   (our 20-run means)
    src/atsa_jssp/paper_table5.py                                 (Sahman 2022 Table 5)
    data/paper_scope_40.txt                                       (the 40-instance scope)
"""
from __future__ import annotations

import pathlib
import statistics
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from atsa_jssp.paper_table5 import TABLE5                          # noqa: E402


def scope_40() -> list[str]:
    lines = (REPO / "data" / "paper_scope_40.txt").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def headline() -> dict:
    scope = scope_40()
    df = pd.concat([pd.read_csv(REPO / "results" / "atsa" / f)
                    for f in ("atsa_ta01.csv", "atsa_ta02_ta80.csv")], ignore_index=True)
    ours = df.groupby("instance")["cmax"].mean()
    Dmap = df.groupby("instance")["D"].first()
    paper = {k: v["ATSA"]["mean"] for k, v in TABLE5.items()}

    rows = []
    for inst in scope:
        o, p, D = float(ours[inst]), float(paper[inst]), int(Dmap[inst])
        diff = (o - p) / p * 100.0                      # signed % difference vs the paper mean
        rows.append((inst, D, o, p, diff))

    diffs = [r[4] for r in rows]
    absd = [abs(d) for d in diffs]
    corr = statistics.correlation([r[1] for r in rows], diffs) if len(rows) > 1 else float("nan")

    return dict(
        n=len(rows),
        max_abs=max(absd),
        mean_signed=sum(diffs) / len(diffs),
        mad=sum(absd) / len(absd),
        within_3_1=all(a <= 3.1 for a in absd),
        corr_D_diff=corr,
        rows=rows,
    )


if __name__ == "__main__":
    h = headline()
    print(f"ATSA reproduction over the paper's {h['n']} instances (data/paper_scope_40.txt):")
    print(f"  max |difference|      : {h['max_abs']:.2f}%")
    print(f"  mean signed difference: {h['mean_signed']:+.2f}%")
    print(f"  mean abs difference   : {h['mad']:.2f}%")
    print(f"  all within +/-3.1%    : {h['within_3_1']}")
    print(f"  corr(D, difference%)  : {h['corr_D_diff']:+.2f}   (near 0 => no size dependence)")
