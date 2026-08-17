#!/usr/bin/env python3
"""
D006 Part A (cont.) -- sampler sensitivity of the job-shop result.

U1 resolved to C3 under the corrected NN start (run_u1_rescore.py). Our entire 40-instance
job-shop run used C1. This re-runs DTSA-core, N=40, seeds 0-19 on ta01/ta11/ta21/ta31/ta41 with
C3 and compares against the C1 means already in results/dtsa_jssp.csv.

Pre-registered reading (written before the run):
  If any per-instance mean moves by more than ~1%, the job-shop result IS sampler-sensitive and
  the full 40-instance run must be repeated with C3; the current numbers become provisional.
  If all five move by <=1%, the job-shop result is sampler-robust and the C1 numbers stand.

Usage:  uv run python -u dtsa/run_sampler_sensitivity.py [--jobs 8]
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
OUT = HERE.parent / "results" / "ablations" / "sampler_sensitivity_c3.csv"
INSTANCES = ["ta01", "ta11", "ta21", "ta31", "ta41"]
SEEDS = list(range(20))


def one(instance: str, seed: int) -> dict:
    jcfg = JSSPConfig(n_trees=40, use_local_search=False, seed_with_mwkr=True,
                      symmetry_sampler="C3")
    row = run_one(instance, seed, jcfg)
    row["config"] = "DTSA-core"
    return row


def done() -> set[str]:
    if not OUT.exists():
        return set()
    counts: dict[str, int] = {}
    with OUT.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts[r["instance"]] = counts.get(r["instance"], 0) + 1
    return {k for k, v in counts.items() if v >= len(SEEDS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.jobs < 1 or args.jobs > 16:
        raise SystemExit("--jobs must be in 1..16.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    finished = done()
    todo = [i for i in INSTANCES if i not in finished]
    print(f"C3 sampler sensitivity: {len(INSTANCES)} instances x {len(SEEDS)} seeds")
    print(f"  to run: {len(todo)}   (--jobs {args.jobs})\n")

    for i, inst in enumerate(todo, 1):
        t0 = time.perf_counter()
        rows = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(one)(inst, s) for s in SEEDS)
        for r in rows:
            assert r["fes_used"] == r["N"] + 6 * r["N"] * r["iters"], r
        new = not OUT.exists()
        with OUT.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerows(rows)
        cm = [r["cmax"] for r in rows]
        print(f"[{i}/{len(todo)}] {inst}  C3 mean {sum(cm)/len(cm):8.2f}  "
              f"min {min(cm)}  {time.perf_counter()-t0:6.1f}s")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
