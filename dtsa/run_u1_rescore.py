#!/usr/bin/env python3
"""
D006 Part A -- re-score the four U1 symmetry samplers under the CORRECTED nearest-neighbour start.

Every earlier U1 score (D002/D003) used the NN tour from city 0. U12 (D004) then showed the paper
almost certainly did NOT start there: best-of-52 accounts for essentially the entire Gate 1
residual (+3.88% -> +0.20%). So every U1 score was computed in a configuration the paper does not
describe, and "no candidate reproduces" may be an artefact of the wrong start city.

THIS IS THE PAPER-FIGURE-MATCHING CONFIGURATION, NOT OUR DEFAULT.
  - best-of-52 NN start (nn_best_start=True)   -- U12, better-performing not more faithful
  - U17 reading (ablation_split_random=True)   -- adopted in D004 as more faithful to Fig. 6
The shipped default stays city-0 + literal, and Gate 1 stays FAILED. This run resolves U1 only;
it does NOT re-open the gate verdict.

Config: symmetry, all three sources {current, random, best}, C1..C4, NS=6, 30 seeds, berlin52.
3 x 4 x 30 = 360 runs. Fast (~2.5 s/run), checkpointed per (source, sampler) group.

Scored on the two pre-registered criteria, against symmetry(current tree):
  (a) deviation of our tsa_mean from TABLE1["symmetry(current tree)"]["tsa_mean"]  (target |dev|<=1.5%)
  (b) our 2-opt gain vs the paper's 1.84%

Usage:  uv run python -u dtsa/run_u1_rescore.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "validation_tsp"))

from joblib import Parallel, delayed              # noqa: E402

from dtsa_reference import Config, check_fe_accounting, dtsa_tsp   # noqa: E402
from tsp import load_tsp                          # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "tsplib"
OUT = HERE.parent / "results" / "ablations" / "u1_rescore.csv"

SOURCES = ["current", "random", "best"]
SAMPLERS = ["C1", "C2", "C3", "C4"]
SEEDS = list(range(30))
N_TREES, MAX_FES, ST = 52, 104_000, 0.5

COLUMNS = ["source", "sampler", "seed", "pre_2opt", "post_2opt", "fes", "iterations",
           "two_opt_evals", "two_opt_moves", "N", "max_fes", "instance",
           "nn_best_start", "ablation_split_random"]


def one(source: str, sampler: str, seed: int) -> dict:
    problem = load_tsp(DATA / "berlin52.tsp", rounded=True)
    cfg = Config(N=N_TREES, max_fes=MAX_FES, ST=ST, seed=seed,
                 ablation=("symmetry", source), t1_seeds_per_row=6,
                 ablation_split_random=True, symmetry_sampler=sampler,
                 two_opt_enabled=True)
    res = dtsa_tsp(problem, cfg, nn_best_start=True)
    check_fe_accounting(res, cfg)
    return {"source": source, "sampler": sampler, "seed": seed,
            "pre_2opt": res.best_pre_2opt, "post_2opt": res.best_post_2opt,
            "fes": res.fes, "iterations": res.iterations,
            "two_opt_evals": res.two_opt_evaluations, "two_opt_moves": res.two_opt_moves,
            "N": N_TREES, "max_fes": MAX_FES, "instance": "berlin52",
            "nn_best_start": True, "ablation_split_random": True}


def done() -> set[tuple[str, str]]:
    if not OUT.exists():
        return set()
    counts: dict[tuple[str, str], int] = {}
    with OUT.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["source"], r["sampler"])
            counts[k] = counts.get(k, 0) + 1
    return {k for k, v in counts.items() if v >= len(SEEDS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.jobs < 1 or args.jobs > 16:
        raise SystemExit("--jobs must be in 1..16.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    finished = done()
    todo = [(s, c) for s in SOURCES for c in SAMPLERS if (s, c) not in finished]
    print(f"U1 re-score: 3 sources x 4 samplers x {len(SEEDS)} seeds = 360 runs")
    print(f"  already complete: {len(finished)} groups   to run: {len(todo)}  (--jobs {args.jobs})\n")

    for i, (src, sampler) in enumerate(todo, 1):
        t0 = time.perf_counter()
        rows = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(one)(src, sampler, s) for s in SEEDS)
        new = not OUT.exists()
        with OUT.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerows(rows)
        pre = sum(r["pre_2opt"] for r in rows) / len(rows)
        post = sum(r["post_2opt"] for r in rows) / len(rows)
        gain = (pre - post) / pre * 100
        print(f"[{i:>2}/{len(todo)}] symmetry({src:<7}) {sampler}  pre {pre:8.2f}  "
              f"2opt-gain {gain:5.2f}%  {time.perf_counter()-t0:6.1f}s")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
