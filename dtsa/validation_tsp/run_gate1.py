#!/usr/bin/env python3
"""
GATE 1 -- reproduce DTSA paper Table 1 (journal p. 885), the operator ablation on BERLIN52.

Design, per the DTSA adaptation notes D5:
  9 configurations  = {swap, shift, symmetry} x {current, random, best} tree
  30 runs each      = seeds 0..29
  berlin52, N = 52, ST = 0.5, MaxFEs = 104000 (D x 2000), rounded EUC_2D (U9)
  U14  -> BOTH seeds-per-tree settings, NS = 6 (literal) and NS = 1
  U1   -> all four samplers C1..C4, on the three SYMMETRY configurations only; the other six
          never invoke symmetry, so a sampler label there would be noise

  36 configuration groups x 30 seeds = 1,080 runs.

CHECKPOINT AND RESUME, in the style of run_tsa_bisect.py. Long runs have been lost to session
teardown and OOM three times on this project (the project design notes §9 item 9, §10). Each configuration
group is appended to the CSV the moment it finishes, and a re-invocation skips whatever is
already there. Killing this script costs at most one group.

--jobs 8. NEVER -1 (the project design notes §10). One job at a time.

Usage:  uv run python dtsa/run_gate1.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from joblib import Parallel, delayed        # noqa: E402

from dtsa_reference import Config, check_fe_accounting, dtsa_tsp   # noqa: E402
from tsp import load_tsp                    # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "tsplib"
OUT = HERE.parent.parent / "results" / "gate1" / "gate1.csv"

OPERATORS = ["swap", "shift", "symmetry"]
SOURCES = ["current", "random", "best"]
NS_SETTINGS = [6, 1]                        # U14: literal first
SAMPLERS = ["C1", "C2", "C3", "C4"]         # U1; symmetry configurations only
SEEDS = list(range(30))                     # §5, p. 883: "run 30 times"

N_TREES = 52
MAX_FES = 104_000
ST = 0.5

COLUMNS = [
    "config", "operator", "source", "NS", "sampler", "seed",
    "pre_2opt", "post_2opt", "fes", "iterations", "two_opt_evals", "two_opt_moves",
    "wall_seconds", "N", "ST", "max_fes", "instance", "rounded", "git_sha",
]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=HERE.parent).stdout.strip()
    except Exception:                       # noqa: BLE001
        return ""


def groups() -> list[tuple[str, str, int, str]]:
    """Every (operator, source, NS, sampler) group that must be run."""
    out = []
    for op in OPERATORS:
        for src in SOURCES:
            for ns in NS_SETTINGS:
                for sampler in (SAMPLERS if op == "symmetry" else ["-"]):
                    out.append((op, src, ns, sampler))
    return out


def group_key(op: str, src: str, ns: int, sampler: str) -> str:
    return f"{op}({src} tree)|NS={ns}|{sampler}"


def one_run(op: str, src: str, ns: int, sampler: str, seed: int, sha: str) -> dict:
    problem = load_tsp(DATA / "berlin52.tsp", rounded=True)
    cfg = Config(
        N=N_TREES, max_fes=MAX_FES, ST=ST, seed=seed,
        ablation=(op, src), t1_seeds_per_row=ns,
        symmetry_sampler=(sampler if sampler != "-" else "C1"),
        two_opt_enabled=True,
    )
    t0 = time.perf_counter()
    res = dtsa_tsp(problem, cfg)
    wall = time.perf_counter() - t0
    check_fe_accounting(res, cfg)           # loud, every run
    return {
        "config": group_key(op, src, ns, sampler),
        "operator": op, "source": src, "NS": ns, "sampler": sampler, "seed": seed,
        "pre_2opt": res.best_pre_2opt, "post_2opt": res.best_post_2opt,
        "fes": res.fes, "iterations": res.iterations,
        "two_opt_evals": res.two_opt_evaluations, "two_opt_moves": res.two_opt_moves,
        "wall_seconds": round(wall, 3),
        "N": N_TREES, "ST": ST, "max_fes": MAX_FES,
        "instance": "berlin52", "rounded": True, "git_sha": sha,
    }


def done_groups() -> set[str]:
    if not OUT.exists():
        return set()
    with OUT.open(newline="", encoding="utf-8") as f:
        seen: dict[str, int] = {}
        for row in csv.DictReader(f):
            seen[row["config"]] = seen.get(row["config"], 0) + 1
    return {k for k, v in seen.items() if v >= len(SEEDS)}


def append(rows: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8, help="worker processes; NEVER -1")
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1. -1 has crashed this machine three times.")

    sha = git_sha()
    todo = [g for g in groups() if group_key(*g) not in done_groups()]
    total = len(groups())
    print(f"Gate 1: {total} configuration groups x {len(SEEDS)} seeds = {total * len(SEEDS)} runs")
    print(f"  already complete : {total - len(todo)}")
    print(f"  to run           : {len(todo)}   (--jobs {args.jobs})\n")

    for i, (op, src, ns, sampler) in enumerate(todo, 1):
        key = group_key(op, src, ns, sampler)
        t0 = time.perf_counter()
        rows = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(one_run)(op, src, ns, sampler, s, sha) for s in SEEDS
        )
        append(rows)                        # checkpoint: one group at a time
        pre = sum(r["pre_2opt"] for r in rows) / len(rows)
        post = sum(r["post_2opt"] for r in rows) / len(rows)
        moved = sum(1 for r in rows if r["two_opt_moves"] > 0)
        print(f"[{i:>2}/{len(todo)}] {key:<34} pre {pre:9.2f}  post {post:9.2f}  "
              f"2opt-improved {moved:>2}/30  {time.perf_counter() - t0:6.1f}s")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
