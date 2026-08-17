#!/usr/bin/env python3
"""
D007 B4 -- does a dispatching-rule seed recover ATSA's initialisation deficit?

Two arms, 5 instances x 20 seeds, ATSA `Config()` untouched (N=40, MaxFEs = D*1000):

  stock  : ATSA exactly as published -- population drawn uniform(-5, 5).
  mwkr1  : the same population with ONE tree replaced by the MWKR dispatching sequence,
           via the P3 `init_pop` parameter.

The random rows are drawn with `np.random.RandomState(seed)`, the same MT19937 the njit
kernel seeds with `np.random.seed(seed)`, so the two arms differ in exactly one tree at
initialisation. (The streams diverge afterwards -- the kernel no longer consumes those
N*D draws -- which is why the comparison is over 20 seeds and not a single run.)

The reading was pre-registered in the project log D007 BEFORE this was run.

Usage:  uv run python -u dtsa/run_b4_atsa_seeded.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from joblib import Parallel, delayed                    # noqa: E402

from atsa_jssp.instance import load_ta                  # noqa: E402  READ-ONLY
from atsa_jssp.atsa import atsa, Config                 # noqa: E402  READ-ONLY apart from P3
from dtsa_jssp import mwkr_sequence, rk_from_sequence   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "ablations" / "b4_atsa_seeded.csv"
INSTANCES = ["ta01", "ta11", "ta21", "ta31", "ta41"]
SEEDS = list(range(20))
ARMS = ["stock", "mwkr1"]
COLUMNS = ["instance", "arm", "seed", "cmax", "fes", "iters", "N", "D", "wall_seconds"]


def one(instance: str, arm: str, seed: int) -> dict:
    inst = load_ta(instance)
    cfg = Config()
    init_pop = None
    if arm == "mwkr1":
        rs = np.random.RandomState(seed)
        init_pop = rs.uniform(cfg.dmin, cfg.dmax, (cfg.N, inst.D))
        init_pop[0] = rk_from_sequence(mwkr_sequence(inst), inst.n)
    t0 = time.perf_counter()
    res = atsa(inst, seed, cfg, init_pop=init_pop)
    return dict(instance=instance, arm=arm, seed=seed, cmax=res["cmax"],
                fes=res["fes"], iters=res["iters"], N=cfg.N, D=inst.D,
                wall_seconds=round(time.perf_counter() - t0, 3))


def done_keys() -> set[tuple[str, str]]:
    if not OUT.exists():
        return set()
    counts: dict[tuple[str, str], int] = {}
    with OUT.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["instance"], r["arm"])
            counts[k] = counts.get(k, 0) + 1
    return {k for k, v in counts.items() if v >= len(SEEDS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys()
    todo = [(i, a) for i in INSTANCES for a in ARMS if (i, a) not in done]
    print(f"B4: {len(INSTANCES)} instances x {len(ARMS)} arms x {len(SEEDS)} seeds")
    print(f"  to run: {len(todo)} groups   (--jobs {args.jobs})\n")

    t_start = time.perf_counter()
    for k, (instance, arm) in enumerate(todo, 1):
        t0 = time.perf_counter()
        rows = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(one)(instance, arm, s) for s in SEEDS)
        new = not OUT.exists()
        with OUT.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerows(rows)
        cm = [r["cmax"] for r in rows]
        print(f"[{k:>2}/{len(todo)}] {instance:<5} {arm:<6} mean {sum(cm)/len(cm):8.1f} "
              f"min {min(cm):>5}   {time.perf_counter()-t0:6.1f}s   "
              f"elapsed {(time.perf_counter()-t_start)/60:5.1f}m")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
