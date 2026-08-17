"""
Run ATSA across the Taillard instances and write per-instance + combined CSVs.

This reproduces the broad ATSA campaign. It runs ta02..ta80 (79 instances, `--runs`
seeded runs each); ta01 is produced separately by the quickstart and lives at
results/atsa/atsa_ta01.csv. The two together cover ta01..ta80.

NOTE ON SCOPE: the source paper's comparison set is only 40 instances (ta01-ta05,
ta11-ta15, ..., ta71-ta75). The extra instances here are additional coverage and are
not part of any paper comparison. See results/atsa/README.md and data/paper_scope_40.txt.

    python run_atsa_campaign.py                 # all logical cores
    python run_atsa_campaign.py --jobs 4        # cap workers
    python run_atsa_campaign.py --runs 20 --jobs 8
"""
from __future__ import annotations
import argparse
import os

# Must precede the numba import (via atsa_jssp.experiment).
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pathlib
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "atsa"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=20, help="seeded runs per instance (seeds 0..runs-1)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="parallel workers (default: logical cores; memory-clamped automatically)")
    args = parser.parse_args()

    from atsa_jssp.experiment import run_one, summarise, git_sha, safe_n_jobs, CSV_COLUMNS
    from atsa_jssp.atsa import Config
    from atsa_jssp.instance import load_ta
    from joblib import Parallel, delayed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [f"ta{k:02d}" for k in range(2, 81)]          # ta02..ta80 (ta01 done by quickstart)
    n_jobs = safe_n_jobs(args.jobs)
    print(f"Running {len(todo)} instances x {args.runs} runs = {len(todo) * args.runs} total runs")
    print(f"Workers: {n_jobs}   Output dir: {OUT_DIR}\n")

    cfg = Config()
    sha = git_sha()
    todo = sorted(todo, key=lambda s: load_ta(s, ROOT / "data/raw").D)   # small instances bank first

    all_rows = []
    t_start = time.perf_counter()
    for name in todo:
        inst = load_ta(name, ROOT / "data/raw")
        t0 = time.perf_counter()
        print(f"  -> {name}  (n={inst.n}, m={inst.m}, D={inst.D})  ...", flush=True)
        tasks = [(name, seed) for seed in range(args.runs)]
        rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(run_one)(n, s, cfg, sha) for n, s in tasks
        )
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        out_path = OUT_DIR / f"atsa_{name}.csv"
        df.to_csv(out_path, index=False)
        s = summarise(df).iloc[0]
        print(f"     mean={s['mean']:.1f}  min={s['min']}  max={s['max']}  std={s['std']:.1f}  "
              f"wall={time.perf_counter() - t0:.0f}s  wrote {out_path}")
        all_rows.extend(rows)

    combined = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
    combined_path = OUT_DIR / "atsa_ta02_ta80.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nDone in {(time.perf_counter() - t_start) / 60:.1f} min.")
    print(f"wrote {combined_path}")
    print(summarise(combined).to_string(index=False))


if __name__ == "__main__":                              # required for multiprocessing on Windows/macOS
    main()
