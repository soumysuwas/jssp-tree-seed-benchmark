#!/usr/bin/env python3
"""
Part B -- the ONE bounded diagnostic of the Gate 1 residual. Persists to CSV so the report can
regenerate its table instead of quoting typed numbers.

2x2 on symmetry(current tree) only, under U17, sampler C1, NS=6, 30 seeds:
    U12 : NN seed from city 0 (literal-ish)  vs  best of all 52 start cities
    F6  : `best` updated after the for-loop (LITERAL, Fig. 6 lines 35-36)  vs  immediately

PRE-REGISTERED STOP RULE (the project log D004, committed at dcdfb68 BEFORE this was run):
    This is the only investigation of the Gate 1 residual. Whatever it shows, the literal
    defaults stay unless a reading is more faithful to Fig. 6. If the residual survives, it is
    reported as unexplained and the workstream proceeds.

NEITHER ARM MAY BECOME THE DEFAULT. Fig. 6 line 5 says "nearest neighbor tour", singular --
best-of-52 is better-performing, not more faithful. Fig. 6 lines 35-36 clearly defer the best
update -- immediate is not more faithful. This run exists to locate the residual, not to remove it.

Usage:  uv run python -u dtsa/run_partb.py [--jobs 8]      # one job at a time
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "validation_tsp"))

from joblib import Parallel, delayed              # noqa: E402

from dtsa_reference import Config, check_fe_accounting, dtsa_tsp    # noqa: E402
from tsp import best_nearest_neighbour_tour, load_tsp, nearest_neighbour_tour  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "tsplib"
OUT = HERE.parent / "results" / "ablations" / "partb_residual.csv"
SEEDS = list(range(30))
COLUMNS = ["u12_nn", "f6_best_update", "seed", "pre_2opt", "post_2opt",
           "two_opt_moves", "fes", "iterations", "nn_seed_length"]


def one(seed: int, nn_best: bool, best_update: str) -> dict:
    p = load_tsp(DATA / "berlin52.tsp", rounded=True)
    cfg = Config(N=52, max_fes=104_000, ST=0.5, seed=seed,
                 ablation=("symmetry", "current"), t1_seeds_per_row=6,
                 symmetry_sampler="C1", two_opt_enabled=True,
                 ablation_split_random=True, best_update=best_update)
    r = dtsa_tsp(p, cfg, nn_best_start=nn_best)
    check_fe_accounting(r, cfg)
    seed_tour = (best_nearest_neighbour_tour(p)[0] if nn_best
                 else nearest_neighbour_tour(p, start=0))
    return {
        "u12_nn": "best_of_52" if nn_best else "city_0",
        "f6_best_update": best_update, "seed": seed,
        "pre_2opt": r.best_pre_2opt, "post_2opt": r.best_post_2opt,
        "two_opt_moves": r.two_opt_moves, "fes": r.fes, "iterations": r.iterations,
        "nn_seed_length": p.tour_length(seed_tour),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for nn_best in (False, True):
        for best_update in ("deferred", "immediate"):
            batch = Parallel(n_jobs=args.jobs, backend="loky")(
                delayed(one)(s, nn_best, best_update) for s in SEEDS)
            rows += batch
            m = sum(r["pre_2opt"] for r in batch) / len(batch)
            print(f"  {'best_of_52' if nn_best else 'city_0':<11} {best_update:<10} "
                  f"pre mean {m:9.2f}")
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
