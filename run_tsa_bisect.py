"""
Task 1 — TSA bisection, crash-resilient.

Writes each instance's 20 runs to results/tsa_bisect.csv AS SOON as that instance finishes,
and skips instances already present on restart. The first attempt at this run buffered all
200 runs in memory and wrote the CSV only at the very end; a system restart at ~114/200
destroyed ~9 minutes of ta71-75 compute for nothing. Resume beats speed here.

Usage:  uv run python run_tsa_bisect.py          # resumes automatically
"""
from __future__ import annotations
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pathlib
import sys

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from atsa_jssp.atsa import Config
from atsa_jssp.experiment import CSV_COLUMNS, git_sha, run_one, safe_n_jobs

OUT = pathlib.Path(__file__).resolve().parent / "results/sweep/tsa_bisect.csv"
# The two ends of the size range: where we match the paper, and where we do not.
INSTANCES = ["ta01", "ta02", "ta03", "ta04", "ta05",
             "ta71", "ta72", "ta73", "ta74", "ta75"]
RUNS = 20

if __name__ == "__main__":
    done: set[str] = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = {i for i, g in prev.groupby("instance") if len(g) >= RUNS}
        print(f"resuming: {sorted(done)} already complete")

    sha = git_sha()
    cfg = Config()
    for name in INSTANCES:
        if name in done:
            continue
        rows = Parallel(n_jobs=safe_n_jobs(-1), backend="loky")(
            delayed(run_one)(name, s, cfg, sha, "TSA") for s in range(RUNS)
        )
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"{name}: mean {df.cmax.mean():8.1f}  min {df.cmax.min()}  "
              f"({df.wall_s.mean():.1f} s/run)  -> appended", flush=True)
    print("TSA BISECT DONE")
