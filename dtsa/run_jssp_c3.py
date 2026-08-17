#!/usr/bin/env python3
"""
The 40-instance job-shop run -- the DTSA adaptation notes D4.

40 Sahman Table 5 instances x 20 seeds (0-19), MaxFEs = D*1000, NS=6, ST=0.5, sampler C3.
D006: sampler C3 is the U1-resolved value (run_u1_rescore.py). This repeats the full run because
run_sampler_sensitivity.py found the job-shop result IS sampler-sensitive (ta31 +1.66% > 1%),
making the C1 numbers provisional. Writes to a SEPARATE CSV; the C1 baseline is left intact.
Both N settings: N=40 (ATSA-matched) and N=D (DTSA-literal).

Each search produces BOTH configurations at no extra cost: `DTSA-core` is the best found by the
search, and `DTSA+LS` is that same solution after one critical-block pass. Two CSV rows per run,
and the local search's evaluations are recorded on the LS row only -- never folded into `fes`
(D3, U7).

⚠️ GATE 1 FAILED. Every figure derived from this CSV must state that on the same page
(the DTSA adaptation notes D5, "Proceeding past a failed Gate 1"). The comparison is to OUR ATSA column
under an identical protocol, not to any published DTSA number -- none exists.

--jobs 8, NEVER -1. Checkpointed and resumable per (instance, N setting): a re-invocation skips
whatever is already in the CSV. Three long runs have been lost to teardown on this project
(the project design notes §9 item 9, §10).

Usage:  uv run python -u dtsa/run_jssp.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from joblib import Parallel, delayed            # noqa: E402

from atsa_jssp.instance import PAPER_INSTANCES  # noqa: E402  READ-ONLY
from dtsa_jssp import CSV_COLUMNS, JSSPConfig, run_one, system_config   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "dtsa" / "dtsa_jssp_c3.csv"
SEEDS = list(range(20))                          # D4: ATSA's protocol, seeds 0-19
N_SETTINGS = [40, None]                          # 40 = ATSA-matched; None = N=D, DTSA-literal


def split_rows(row: dict) -> list[dict]:
    """One search -> a DTSA-core row and a DTSA+LS row. Both are exact, not estimates."""
    core = dict(row)
    core["config"] = "DTSA-core"
    core["cmax"] = row["pre_local_search_cmax"]
    core["local_search_evals"] = 0
    core["local_search_moves"] = 0
    ls = dict(row)
    ls["config"] = "DTSA+LS"
    return [core, ls]


def one(instance: str, seed: int, n_trees):
    jcfg = JSSPConfig(n_trees=n_trees, use_local_search=True, symmetry_sampler="C3")
    return split_rows(run_one(instance, seed, jcfg))


def done_keys() -> set[tuple[str, str]]:
    if not OUT.exists():
        return set()
    counts: dict[tuple[str, str], int] = {}
    with OUT.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["instance"], r["n_setting"])
            counts[k] = counts.get(k, 0) + 1
    return {k for k, v in counts.items() if v >= len(SEEDS) * 2}     # 2 rows per run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1. -1 has crashed this machine three times.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "dtsa_jssp_system_config.json").write_text(
        json.dumps(system_config(), indent=2), encoding="utf-8")

    done = done_keys()
    todo = [(inst, n) for n in N_SETTINGS for inst in PAPER_INSTANCES
            if (inst, "N=D" if n is None else f"N={n}") not in done]
    print(f"job-shop run: {len(PAPER_INSTANCES)} instances x {len(SEEDS)} seeds x "
          f"{len(N_SETTINGS)} N settings")
    print(f"  already complete : {len(PAPER_INSTANCES) * len(N_SETTINGS) - len(todo)} groups")
    print(f"  to run           : {len(todo)} groups   (--jobs {args.jobs})\n")

    t_start = time.perf_counter()
    for i, (inst, n_trees) in enumerate(todo, 1):
        t0 = time.perf_counter()
        batches = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(one)(inst, s, n_trees) for s in SEEDS)
        rows = [r for b in batches for r in b]
        for r in rows:                                   # the identity, on every row
            assert r["fes_used"] == r["N"] + 6 * r["N"] * r["iters"], r
        new = not OUT.exists()
        with OUT.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerows(rows)
        core = [r["cmax"] for r in rows if r["config"] == "DTSA-core"]
        ls = [r["cmax"] for r in rows if r["config"] == "DTSA+LS"]
        label = "N=D" if n_trees is None else f"N={n_trees}"
        el = time.perf_counter() - t_start
        print(f"[{i:>2}/{len(todo)}] {inst:<5} {label:<5} core mean {sum(core)/len(core):8.1f} "
              f"min {min(core):>5}   LS mean {sum(ls)/len(ls):8.1f} min {min(ls):>5}   "
              f"{time.perf_counter()-t0:6.1f}s   elapsed {el/3600:4.2f}h")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
