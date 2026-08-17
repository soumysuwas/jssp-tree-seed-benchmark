#!/usr/bin/env python3
"""
D007 Part B -- honest timing. This is the Table 6 column.

Every `wall_seconds` in the existing job-shop CSVs was recorded at varying `--jobs`
(8, then 12, then 7, then 6). Those makespans are unaffected -- each run is an independent
single-threaded process with its own seed, so a run is bit-identical at any job count --
but the TIMINGS are contaminated by inter-process contention and are not usable.

Two sub-commands:

  serial     B1. All 40 instances x 3 seeds (0-2), N=40, DTSA-core, sampler C3, --jobs 1,
             nothing else on the machine. -> results/timing_serial.csv
             B3 (throughput, evals/sec) is read off the same rows.

  contention B2. Eight instances spanning the size range, 1 seed each, run twice: once
             strictly sequential, once with all eight in flight at --jobs 8.
             -> results/timing_contention.csv

Local search is OFF: DTSA-core is the configuration every reported number uses.
Checkpointed per instance.

Usage:  uv run python -u dtsa/run_timing.py serial
        uv run python -u dtsa/run_timing.py contention
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from joblib import Parallel, delayed                    # noqa: E402

from atsa_jssp.instance import PAPER_INSTANCES          # noqa: E402  READ-ONLY
from dtsa_jssp import JSSPConfig, run_one               # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SERIAL_OUT = HERE.parent / "results" / "timing" / "timing_serial.csv"
CONTENTION_OUT = HERE.parent / "results" / "timing" / "timing_contention.csv"
SERIAL_SEEDS = [0, 1, 2]
CONTENTION_INSTANCES = ["ta01", "ta11", "ta21", "ta31", "ta41", "ta51", "ta61", "ta71"]
COLUMNS = ["instance", "seed", "jobs", "cmax", "fes_used", "iters", "N", "D",
           "wall_seconds", "evals_per_second"]


def measure(instance: str, seed: int, jobs: int) -> dict:
    """One DTSA-core search, timed. `jobs` is recorded, not used -- the caller sets concurrency."""
    jcfg = JSSPConfig(n_trees=40, use_local_search=False, seed_with_mwkr=True,
                      symmetry_sampler="C3")
    t0 = time.perf_counter()
    row = run_one(instance, seed, jcfg)
    wall = time.perf_counter() - t0
    assert row["fes_used"] == row["N"] + 6 * row["N"] * row["iters"], row
    return dict(instance=instance, seed=seed, jobs=jobs,
                cmax=row["pre_local_search_cmax"], fes_used=row["fes_used"],
                iters=row["iters"], N=row["N"], D=row["D"],
                wall_seconds=round(wall, 3),
                evals_per_second=round(row["fes_used"] / wall, 1))


def append(out: pathlib.Path, rows: list[dict]) -> None:
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def done(out: pathlib.Path, key: str, need: int) -> set:
    if not out.exists():
        return set()
    counts: dict = {}
    with out.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["instance"], r[key])
            counts[k] = counts.get(k, 0) + 1
    return {k for k, v in counts.items() if v >= need}


def serial() -> None:
    SERIAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    have = {i for (i, _j) in done(SERIAL_OUT, "jobs", len(SERIAL_SEEDS))}
    todo = [i for i in PAPER_INSTANCES if i not in have]
    print(f"B1 serial timing: {len(todo)} instances x {len(SERIAL_SEEDS)} seeds, --jobs 1\n")
    t_start = time.perf_counter()
    for k, instance in enumerate(todo, 1):
        rows = [measure(instance, s, 1) for s in SERIAL_SEEDS]
        append(SERIAL_OUT, rows)
        mean = sum(r["wall_seconds"] for r in rows) / len(rows)
        eps = sum(r["evals_per_second"] for r in rows) / len(rows)
        print(f"[{k:>2}/{len(todo)}] {instance:<5} D={rows[0]['D']:<4} "
              f"mean {mean:8.2f}s  {eps:9.0f} evals/s   "
              f"elapsed {(time.perf_counter()-t_start)/60:6.1f}m")
    print(f"\nwrote {SERIAL_OUT}")


def contention() -> None:
    CONTENTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    have = done(CONTENTION_OUT, "jobs", 1)
    t_start = time.perf_counter()

    todo1 = [i for i in CONTENTION_INSTANCES if (i, "1") not in have]
    print(f"B2a sequential: {len(todo1)} instances, one at a time")
    for k, instance in enumerate(todo1, 1):
        row = measure(instance, 0, 1)
        append(CONTENTION_OUT, [row])
        print(f"[{k}/{len(todo1)}] {instance:<5} {row['wall_seconds']:8.2f}s  "
              f"elapsed {(time.perf_counter()-t_start)/60:5.1f}m")

    todo8 = [i for i in CONTENTION_INSTANCES if (i, "8") not in have]
    if todo8:
        print(f"\nB2b concurrent: {len(todo8)} instances, all in flight at --jobs 8")
        rows = Parallel(n_jobs=8, backend="loky")(
            delayed(measure)(i, 0, 8) for i in todo8)
        append(CONTENTION_OUT, rows)
        for r in rows:
            print(f"  {r['instance']:<5} {r['wall_seconds']:8.2f}s")
    print(f"\nwrote {CONTENTION_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serial", "contention"])
    args = ap.parse_args()
    (serial if args.mode == "serial" else contention)()
