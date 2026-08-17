#!/usr/bin/env python3
"""
The MWKR-seeding ablation -- the run that can kill the headline result.

DTSA seeds tree 1 with an MWKR dispatching solution (the DTSA adaptation notes D2). ATSA seeds every
tree randomly. If DTSA's advantage over ATSA is really the seed rather than the search, this
shows it.

DTSA-core, N=40, MaxFEs = D*1000, seeds 0-19, on ta01/ta11/ta21/ta31/ta41, seeding ON and OFF.
5 x 2 x 20 = 200 runs.

The reading was PRE-REGISTERED in the project log D005 at commit 28cc575, before this was run:
    gap_unseeded >= 0.5 * gap_seeded  ->  the win is about the SEARCH
    gap_unseeded <  0.5 * gap_seeded  ->  the win is about INITIALISATION, a different claim

Usage:  uv run python -u dtsa/run_mwkr_ablation.py [--jobs 12]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from joblib import Parallel, delayed              # noqa: E402

from dtsa_jssp import CSV_COLUMNS, JSSPConfig, run_one   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "ablations" / "mwkr_ablation.csv"
INSTANCES = ["ta01", "ta11", "ta21", "ta31", "ta41"]
SEEDS = list(range(20))


def one(instance: str, seed: int, mwkr: bool) -> dict:
    jcfg = JSSPConfig(n_trees=40, use_local_search=False, seed_with_mwkr=mwkr,
                      symmetry_sampler="C1")
    row = run_one(instance, seed, jcfg)
    row["config"] = "DTSA-core"
    return row


def done() -> set[tuple[str, str]]:
    if not OUT.exists():
        return set()
    counts: dict[tuple[str, str], int] = {}
    with OUT.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["instance"], r["seeded_with_mwkr"])
            counts[k] = counts.get(k, 0) + 1
    return {k for k, v in counts.items() if v >= len(SEEDS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    if args.jobs < 1 or args.jobs > 16:
        raise SystemExit("--jobs must be in 1..16 (hard cap 16 on this machine).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    finished = done()
    todo = [(i, m) for m in (True, False) for i in INSTANCES
            if (i, str(m)) not in finished]
    print(f"MWKR ablation: {len(INSTANCES)} instances x 2 x {len(SEEDS)} seeds "
          f"= {len(INSTANCES) * 2 * len(SEEDS)} runs")
    print(f"  to run: {len(todo)} groups   (--jobs {args.jobs})\n")

    for i, (inst, mwkr) in enumerate(todo, 1):
        t0 = time.perf_counter()
        rows = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(one)(inst, s, mwkr) for s in SEEDS)
        for r in rows:
            assert r["fes_used"] == r["N"] + 6 * r["N"] * r["iters"], r
        new = not OUT.exists()
        with OUT.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerows(rows)
        cm = [r["cmax"] for r in rows]
        print(f"[{i:>2}/{len(todo)}] {inst:<5} mwkr={str(mwkr):<5} "
              f"mean {sum(cm)/len(cm):8.1f}  min {min(cm):>5}  "
              f"{time.perf_counter()-t0:6.1f}s")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
