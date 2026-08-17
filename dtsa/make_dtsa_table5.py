#!/usr/bin/env python3
"""
Regenerate results/dtsa/dtsa_table5.csv from results/dtsa/dtsa_jssp_c3.csv.

This is the per-instance summary of the DTSA-core job-shop runs, in the shape of
Sahman (2022) Table 5 (one row per instance x N-setting: runs/mean/median/min/max/std).
It is a descriptive summary prepared as input for any separate statistical analysis;
it is not itself a statistical test.

    python dtsa/make_dtsa_table5.py

The aggregation logic is a verbatim lift of the routine that originally produced the
committed CSV, with the file paths pointed at this repository's results/ layout.
"""
from __future__ import annotations

import pathlib

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "results" / "dtsa" / "dtsa_jssp_c3.csv"
OUT = REPO / "results" / "dtsa" / "dtsa_table5.csv"

COLUMNS = ["instance", "n", "m", "D", "runs", "mean", "med", "min", "max", "std",
           "algorithm", "N_setting", "N", "sampler", "max_fes", "seeds"]


def table5() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    core = df[df.config == "DTSA-core"]
    rows = []
    for (inst, ns), g in core.groupby(["instance", "n_setting"]):
        c = g["cmax"]
        rows.append({
            "instance": inst, "n": int(g["n"].iloc[0]), "m": int(g["m"].iloc[0]),
            "D": int(g["D"].iloc[0]), "runs": int(c.count()),
            "mean": round(c.mean(), 2), "med": c.median(), "min": int(c.min()),
            "max": int(c.max()), "std": round(c.std(), 2),
            "algorithm": "DTSA-core", "N_setting": ns, "N": int(g["N"].iloc[0]),
            "sampler": g["sampler"].iloc[0], "max_fes": int(g["max_fes"].iloc[0]),
            "seeds": ",".join(str(s) for s in sorted(g["seed"].unique())),
        })
    out = pd.DataFrame(rows, columns=COLUMNS).sort_values(["N_setting", "instance"])
    out.to_csv(OUT, index=False)
    return out


if __name__ == "__main__":
    t = table5()
    print(f"wrote {OUT}  ({len(t)} rows)")
