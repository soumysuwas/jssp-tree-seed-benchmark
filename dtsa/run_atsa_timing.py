#!/usr/bin/env python3
"""
D011 -- ATSA runtime, measured the SAME WAY DTSA's §5.6 timing was.

A runtime question was raised in review: is DTSA faster than ATSA at equal budget?
To answer fairly, ATSA must be timed under the identical protocol DTSA was: stock ATSA
(`Config()` untouched), N=40, MaxFEs = D*1000, seeds 0-2, all 40 Taillard instances,
`--jobs 1`, elapsed seconds, nothing else running. -> results/atsa_timing_serial.csv

CORRECTNESS GATE (asserted before any timing is trusted): this run re-executes ATSA only
to TIME it. The makespans it produces MUST be bit-identical to our validated ATSA results
(results/atsa/atsa_ta01.csv + results/atsa/atsa_ta02_ta80.csv -- the canonical pair metrics.py uses)
for every (instance, seed). If a single one differs, the baseline moved and we STOP. The
entry point and seed convention are exactly the validated run's (`atsa(inst, seed, Config())`);
no new code path.

JIT note: ATSA's search kernel is njit-compiled; the first call in a process pays a one-time
compile. DTSA's loop is plain NumPy with no such cost, so to compare steady-state execution
we warm the JIT once (a discarded ta01 run) before timing. Every recorded second is then
pure execution, matching what DTSA's timing measured.

Contention (Part B): the same 8 instances DTSA used, seed 0, at `--jobs 1`, `--jobs 8`, and
-- since ATSA has never hit the RAM pressure DTSA did -- additionally `--jobs 24` (this
machine's logical-core count). The 24-way arm launches 24 real concurrent tasks
(8 instances x seeds 0-2) so the cores are genuinely saturated; the seed-0 row is the one
compared. -> results/atsa_timing_contention.csv

Usage:  uv run python -u dtsa/run_atsa_timing.py serial
        uv run python -u dtsa/run_atsa_timing.py contention
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from joblib import Parallel, delayed                    # noqa: E402

from atsa_jssp.instance import PAPER_INSTANCES, load_ta  # noqa: E402  READ-ONLY
from atsa_jssp.atsa import atsa, Config                  # noqa: E402  READ-ONLY (no edit)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SERIAL_OUT = HERE.parent / "results" / "timing" / "atsa_timing_serial.csv"
CONTENTION_OUT = HERE.parent / "results" / "timing" / "atsa_timing_contention.csv"
VALIDATED = ["results/atsa/atsa_ta01.csv", "results/atsa/atsa_all_instances.csv"]
SERIAL_SEEDS = [0, 1, 2]
CONTENTION_INSTANCES = ["ta01", "ta11", "ta21", "ta31", "ta41", "ta51", "ta61", "ta71"]
SERIAL_COLUMNS = ["instance", "seed", "jobs", "cmax", "fes_used", "iters", "N", "D",
                  "wall_seconds", "evals_per_second"]
CONTENTION_COLUMNS = ["instance", "seed", "jobs", "cmax", "D", "wall_seconds"]


def validated_cmax() -> dict[tuple[str, int], int]:
    """Per-(instance, seed) makespans from the validated ATSA CSVs -- the gate baseline."""
    out: dict[tuple[str, int], int] = {}
    for rel in VALIDATED:
        p = ROOT / rel
        if not p.exists():
            continue
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[(r["instance"], int(r["seed"]))] = int(round(float(r["cmax"])))
    if not out:
        raise SystemExit("No validated ATSA CSVs found; cannot run the makespan gate.")
    return out


def run_atsa(instance: str, seed: int) -> tuple[dict, float]:
    """One stock-ATSA search, timed. Config() untouched -- the validated code path."""
    inst = load_ta(instance)
    cfg = Config()
    t0 = time.perf_counter()
    res = atsa(inst, seed, cfg)
    wall = time.perf_counter() - t0
    row = dict(instance=instance, seed=seed, cmax=int(res["cmax"]), fes_used=int(res["fes"]),
               iters=int(res["iters"]), N=cfg.N, D=inst.D, wall_seconds=round(wall, 3),
               evals_per_second=round(res["fes"] / wall, 1))
    return row, wall


def append(out: pathlib.Path, cols: list[str], rows: list[dict]) -> None:
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def _done_serial() -> set[str]:
    if not SERIAL_OUT.exists():
        return set()
    counts: dict[str, int] = {}
    with SERIAL_OUT.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts[r["instance"]] = counts.get(r["instance"], 0) + 1
    return {i for i, c in counts.items() if c >= len(SERIAL_SEEDS)}


def _assert_gate(base: dict[tuple[str, int], int]) -> None:
    """Read back every serial row and assert its makespan equals the validated one."""
    checked, missing, bad = 0, [], []
    with SERIAL_OUT.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["instance"], int(r["seed"]))
            if key not in base:
                missing.append(key)
                continue
            if int(r["cmax"]) != base[key]:
                bad.append((key, int(r["cmax"]), base[key]))
            checked += 1
    if bad:
        print("\n*** MAKESPAN GATE FAILED -- the re-run ATSA did NOT reproduce the validated "
              "makespans. The comparison baseline has moved; STOP. ***")
        for key, got, exp in bad[:20]:
            print(f"    {key}: re-run {got}  != validated {exp}")
        raise SystemExit(1)
    print(f"\nMAKESPAN GATE PASS: {checked}/{checked} re-run makespans bit-identical to the "
          f"validated ATSA. Overlap seeds: {sorted(SERIAL_SEEDS)} (validated has 20 seeds/instance).")
    if missing:
        print(f"  note: {len(missing)} rows had no validated counterpart (unexpected): {missing[:5]}")


def serial() -> None:
    SERIAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    base = validated_cmax()
    have = _done_serial()
    todo = [i for i in PAPER_INSTANCES if i not in have]
    print(f"ATSA serial timing: {len(todo)} instances x {len(SERIAL_SEEDS)} seeds, --jobs 1")
    if todo:
        print("warming njit (discarded)…", flush=True)
        run_atsa("ta01", 0)                              # compile once; not recorded
    t_start = time.perf_counter()
    for k, instance in enumerate(todo, 1):
        rows = []
        for s in SERIAL_SEEDS:
            row, _ = run_atsa(instance, s)
            row["jobs"] = 1
            exp = base.get((instance, s))
            flag = "OK" if exp is not None and row["cmax"] == exp else "MISMATCH!"
            if flag != "OK":
                print(f"    {instance} seed {s}: re-run {row['cmax']} vs validated {exp}  <-- {flag}")
            rows.append(row)
        append(SERIAL_OUT, SERIAL_COLUMNS, rows)
        mean = sum(r["wall_seconds"] for r in rows) / len(rows)
        print(f"[{k:>2}/{len(todo)}] {instance:<5} D={rows[0]['D']:<4} mean {mean:8.2f}s   "
              f"elapsed {(time.perf_counter()-t_start)/60:6.1f}m", flush=True)
    _assert_gate(base)
    print(f"wrote {SERIAL_OUT}")


def _done_contention() -> set[tuple[str, int]]:
    if not CONTENTION_OUT.exists():
        return set()
    out = set()
    with CONTENTION_OUT.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.add((r["instance"], int(r["jobs"])))
    return out


def _crow(instance: str, seed: int, jobs: int) -> dict:
    row, _ = run_atsa(instance, seed)
    return dict(instance=instance, seed=seed, jobs=jobs, cmax=row["cmax"], D=row["D"],
                wall_seconds=row["wall_seconds"])


def contention() -> None:
    CONTENTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    have = _done_contention()
    print("warming njit (discarded)…", flush=True)
    run_atsa("ta01", 0)

    # --jobs 1: strictly sequential, seed 0.
    todo1 = [i for i in CONTENTION_INSTANCES if (i, 1) not in have]
    if todo1:
        print(f"jobs=1 sequential: {len(todo1)} instances")
        rows = [_crow(i, 0, 1) for i in todo1]
        append(CONTENTION_OUT, CONTENTION_COLUMNS, rows)
        for r in rows:
            print(f"  {r['instance']:<5} {r['wall_seconds']:8.2f}s")

    # --jobs 8: the 8 instances, seed 0, all in flight.
    todo8 = [i for i in CONTENTION_INSTANCES if (i, 8) not in have]
    if todo8:
        print(f"\njobs=8 concurrent: {len(todo8)} instances in flight")
        rows = Parallel(n_jobs=8, backend="loky")(delayed(_crow)(i, 0, 8) for i in todo8)
        append(CONTENTION_OUT, CONTENTION_COLUMNS, rows)
        for r in rows:
            print(f"  {r['instance']:<5} {r['wall_seconds']:8.2f}s")

    # --jobs 24: 24 real concurrent tasks (8 instances x seeds 0-2) to saturate all cores;
    # the seed-0 row per instance is the one compared. Back off on memory pressure.
    todo24 = [i for i in CONTENTION_INSTANCES if (i, 24) not in have]
    if todo24:
        jobs = 24
        while jobs >= 8:
            print(f"\njobs={jobs}: {len(todo24)}x3 = {len(todo24)*3} tasks in flight")
            try:
                rows = Parallel(n_jobs=jobs, backend="loky")(
                    delayed(_crow)(i, s, jobs) for i in todo24 for s in (0, 1, 2))
                append(CONTENTION_OUT, CONTENTION_COLUMNS, rows)
                for r in rows:
                    if r["seed"] == 0:
                        print(f"  {r['instance']:<5} seed0 {r['wall_seconds']:8.2f}s")
                print(f"  (ran clean at n_jobs={jobs})")
                break
            except MemoryError:
                jobs //= 2
                print(f"  MemoryError -- backing off to n_jobs={jobs}")
    print(f"\nwrote {CONTENTION_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serial", "contention"])
    args = ap.parse_args()
    (serial if args.mode == "serial" else contention)()
