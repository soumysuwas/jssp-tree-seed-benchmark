#!/usr/bin/env python3
"""
D006 Part B2 -- why the N5 local search finds almost nothing. Timeboxed diagnostic, no fix.

Run DTSA-core to convergence on ta01 and ta71 (mirroring run_one exactly), take the best
solution, and on that converged schedule measure: critical-path length, the FULL block-size
distribution (NOT filtered to len>=2), and how many N5 candidate moves exist. Decide:
  genuine : the critical path fragments into length-1 machine-blocks, N5 is near-useless, code ok.
  bug     : blocks of length >= 2 exist on the path and are not being surfaced.

Run from the repo root:  uv run python -u dtsa/diag_localsearch.py
"""
from __future__ import annotations

import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dtsa_jssp import (Config, RK_LO, RK_HI, _critical_blocks, _schedule,   # noqa: E402
                       dtsa, evaluate_fast, load_ta, mwkr_sequence,
                       n5_local_search, rk_from_sequence, rk_to_job_sequence)


def full_path_blocks(seq, inst):
    """Critical path split into maximal same-machine runs, WITHOUT the len>=2 filter."""
    cmax, starts, ends, machine_order = _schedule(seq, inst)
    pos_on_machine = {}
    for mach, ops_on_m in enumerate(machine_order):
        for idx, jk in enumerate(ops_on_m):
            pos_on_machine[jk] = (mach, idx)
    cur = None
    for j in range(inst.n):
        if int(ends[j, inst.m - 1]) == cmax:
            cur = (j, inst.m - 1)
            break
    path = [cur]
    while True:
        j, k = cur
        st = int(starts[j, k])
        if k > 0 and int(ends[j, k - 1]) == st:
            cur = (j, k - 1)
        else:
            mach, idx = pos_on_machine[(j, k)]
            if idx == 0:
                break
            pj, pk = machine_order[mach][idx - 1]
            if int(ends[pj, pk]) != st:
                break
            cur = (pj, pk)
        path.append(cur)
    path.reverse()
    blocks, block = [], [path[0]]
    for prev, node in zip(path, path[1:]):
        same = int(inst.route[prev[0], prev[1]]) == int(inst.route[node[0], node[1]])
        if same:
            block.append(node)
        else:
            blocks.append(block)
            block = [node]
    blocks.append(block)
    return cmax, path, blocks


def converged_vector(instance: str, seed: int = 0) -> tuple:
    """Reproduce run_one's DTSA-core search and return (inst, best_vector)."""
    inst = load_ta(instance)
    D = inst.D
    route, ptime = inst.arrays()
    rng = np.random.default_rng(seed)
    pop = rng.uniform(RK_LO, RK_HI, (40, D))
    pop[0] = rk_from_sequence(mwkr_sequence(inst), inst.n)
    cfg = Config(N=40, max_fes=D * 1000, ST=0.5, NS=6, seed=seed,
                 symmetry_sampler="C1", two_opt_enabled=False)

    def evaluate(x):
        return float(evaluate_fast(x, route, ptime, inst.n, inst.m))

    res = dtsa(evaluate, pop, cfg)
    return inst, res.best_vector


if __name__ == "__main__":
    import csv
    rows = []
    for instance in ["ta01", "ta71"]:
        inst, x = converged_vector(instance)
        seq = rk_to_job_sequence(x, inst.n).astype(np.int32)
        cmax, path, blocks = full_path_blocks(seq, inst)
        dist = collections.Counter(len(b) for b in blocks)
        ge2 = [b for b in blocks if len(b) >= 2]
        n5_moves = sum(1 + (1 if len(b) >= 3 else 0) for b in ge2)
        shipped = _critical_blocks(seq, inst)
        _, cmax_ls, evals, moves = n5_local_search(x, inst)

        print(f"\n===== {instance}  ({inst.n}x{inst.m}, D={inst.D})  converged Cmax={cmax} =====")
        print(f"  critical-path length (operations)  : {len(path)}")
        print(f"  blocks on path (all machine-runs)  : {len(blocks)}")
        print(f"  block-size distribution            : {dict(sorted(dist.items()))}")
        print(f"  blocks of length >= 2              : {len(ge2)}")
        print(f"  N5 candidate moves available       : {n5_moves}")
        print(f"  _critical_blocks() returns (len>=2): {len(shipped)}")
        print(f"  n5_local_search: evals={evals} moves={moves}  Cmax {cmax}->{cmax_ls}")

        rows.append(dict(instance=instance, D=inst.D, cmax=cmax,
                         path_ops=len(path), blocks=len(blocks),
                         block_sizes=str(dict(sorted(dist.items()))),
                         blocks_ge2=len(ge2), n5_candidates=n5_moves,
                         ls_evals=evals, ls_moves=moves, cmax_after=cmax_ls))

    out = pathlib.Path(__file__).resolve().parent.parent / "results" / "ablations" / "n5_diag.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
