"""
DTSA for the Job Shop Scheduling Problem -- the port specified in the DTSA adaptation notes D1-D4.

WRITTEN, NOT VALIDATED. Nothing here may be run on the 40-instance set until Gate 1 passes
(D5). A DTSA job-shop number from an unvalidated implementation is unfalsifiable -- that is the
whole reason the TSP gate exists.

The search kernel is NOT reimplemented: `dtsa_reference.dtsa` is reused verbatim. It takes an
`evaluate()` and a population and knows nothing about TSP, so the port is exactly three things --
an objective, an initial population, and a local search:

  D1  representation   random keys in [-5,5]^D, D = n*m, decoded by the VERIFIED, READ-ONLY
                       `atsa_jssp.decoder.evaluate_fast`. Licensed by the equivariance proof
                       (the project design notes §8 D4), which is re-asserted for DTSA's own operators in
                       dtsa/tests/test_operators.py::test_d1_equivariance_rk_vs_sequence.
  D2  initialisation   tree 0 = MWKR dispatching order; trees 1..N-1 = uniform random, exactly
                       as ATSA initialises (`np.random.uniform(dmin, dmax, (N, D))`).
  D3  local search     two configurations. DTSA-core has none. DTSA+LS runs a critical-block
                       (N5) search ONCE, after termination, on `best` only, and its evaluations
                       are counted SEPARATELY and never folded into `fes`.
  D4  parameters       MaxFEs = D*1000, 20 runs (seeds 0-19), NS=6, ST=0.5, N a parameter so
                       both N=D and N=40 can be run. The N=D caveat in D4 applies to any
                       reporting of that arm.

NOTHING IN `src/atsa_jssp/` IS MODIFIED. Three symbols are imported read-only:
`evaluate_fast`, `rk_to_job_sequence`, `load_ta`.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from atsa_jssp.decoder import evaluate_fast, rk_to_job_sequence      # noqa: E402  READ-ONLY
from atsa_jssp.instance import PAPER_INSTANCES, load_ta              # noqa: E402  READ-ONLY
from dtsa_reference import Config, check_fe_accounting, dtsa         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RK_LO, RK_HI = -5.0, 5.0                     # ATSA Table 4's range; kept identical for fairness


# ==============================================================================================
# D2 -- MWKR dispatching seed, and the exact sequence <-> random-key round trip
# ==============================================================================================
def mwkr_sequence(inst) -> np.ndarray:
    """
    Most Work Remaining, as a JOB ORDERING.

    At each of the D steps, among jobs that still have unscheduled operations, take the one with
    the greatest total remaining processing time; ties by lowest job index, so it is
    deterministic. Emits a job sequence (each job id exactly m times) -- the same object
    `rk_to_job_sequence` produces -- which the verified semi-active decoder then turns into a
    schedule.

    ⚠️ OUR CONSTRUCTION, and a specific reading. A textbook MWKR *dispatching rule* schedules
    operation by operation against machine availability; this orders jobs and lets the decoder
    place them. The sequence-based form is the one that fits this representation, and it is the
    structural analogue of DTSA's nearest-neighbour tour: one greedy, problem-aware constructive
    pass, no search. the DTSA adaptation notes D2 lists SPT/LPT/none as the alternatives.
    """
    n, m = inst.n, inst.m
    ptime = inst.ptime
    next_op = np.zeros(n, dtype=np.int64)
    remaining = ptime.sum(axis=1).astype(np.int64)

    seq = np.empty(n * m, dtype=np.int32)
    for t in range(n * m):
        best_j, best_r = -1, -1
        for j in range(n):
            if next_op[j] < m and remaining[j] > best_r:
                best_j, best_r = j, int(remaining[j])
        seq[t] = best_j
        remaining[best_j] -= int(ptime[best_j, next_op[best_j]])
        next_op[best_j] += 1
    return seq


def rk_from_sequence(seq: np.ndarray, n: int) -> np.ndarray:
    """
    Exact inverse of `rk_to_job_sequence`: build a random-key vector that decodes to `seq`.

    The decoder assigns 1-based ranks tau to positions and maps job = tau % n. The ranks 1..D
    hit each residue class mod n exactly m times, and `seq` contains each job exactly m times,
    so an assignment always exists: give the positions holding job c the ranks congruent to c,
    in increasing order. Then map ranks to strictly increasing values in [RK_LO, RK_HI].

    D2a: the round trip is asserted for every Taillard instance by
    `check_roundtrip_all_instances()` before any seeding is trusted.
    """
    D = seq.shape[0]
    ranks_for = {c: [] for c in range(n)}
    for r in range(1, D + 1):
        ranks_for[r % n].append(r)

    tau = np.empty(D, dtype=np.int64)
    cursor = {c: 0 for c in range(n)}
    for pos in range(D):
        c = int(seq[pos])
        tau[pos] = ranks_for[c][cursor[c]]
        cursor[c] += 1

    return RK_LO + (RK_HI - RK_LO) * (tau - 1) / (D - 1)


def check_roundtrip(inst) -> bool:
    """D2a for one instance: sequence -> RK -> sequence must be the identity."""
    seq = mwkr_sequence(inst)
    return bool(np.array_equal(rk_to_job_sequence(rk_from_sequence(seq, inst.n), inst.n), seq))


def check_roundtrip_all_instances(names: list[str] | None = None) -> dict[str, bool]:
    """D2a across the whole benchmark. Must be all-True before MWKR seeding is used."""
    return {name: check_roundtrip(load_ta(name)) for name in (names or PAPER_INSTANCES)}


# ==============================================================================================
# D3 -- critical-block (N5) local search. The job-shop analogue of DTSA's 2-opt.
# ==============================================================================================
def _schedule(seq: np.ndarray, inst):
    """
    Semi-active decode that ALSO returns each machine's processing order.

    This duplicates `atsa_jssp.decoder.build_schedule`'s rule because the local search needs the
    machine orders to find critical blocks, and that module is read-only. The duplication is
    guarded: `tests/test_dtsa_jssp.py::test_local_schedule_matches_the_verified_decoder` asserts
    this makespan equals `evaluate_fast`'s on random inputs across instances. A second copy of
    decode logic is exactly the kind of thing that has bitten this project before, so it is
    checked rather than trusted.
    """
    n, m = inst.n, inst.m
    machine_free = np.zeros(m, dtype=np.int64)
    job_ready = np.zeros(n, dtype=np.int64)
    op_idx = np.zeros(n, dtype=np.int64)
    starts = np.zeros((n, m), dtype=np.int64)
    ends = np.zeros((n, m), dtype=np.int64)
    machine_order: list[list[tuple[int, int]]] = [[] for _ in range(m)]

    for job in seq:
        job = int(job)
        k = int(op_idx[job])
        mach = int(inst.route[job, k])
        dur = int(inst.ptime[job, k])
        st = max(int(machine_free[mach]), int(job_ready[job]))
        starts[job, k] = st
        ends[job, k] = st + dur
        machine_free[mach] = st + dur
        job_ready[job] = st + dur
        machine_order[mach].append((job, k))
        op_idx[job] += 1

    return int(job_ready.max()), starts, ends, machine_order


def _critical_blocks(seq: np.ndarray, inst) -> list[list[tuple[int, int]]]:
    """
    One critical path, split into maximal same-machine blocks.

    Backtrack from an operation finishing at Cmax: its predecessor is either the same job's
    previous operation or the previous operation on its machine, whichever ends exactly when it
    starts. Job predecessor is tried first, so the path is deterministic.
    """
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
    if cur is None:
        return []

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
        same_machine = int(inst.route[prev[0], prev[1]]) == int(inst.route[node[0], node[1]])
        if same_machine:
            block.append(node)
        else:
            blocks.append(block)
            block = [node]
    blocks.append(block)
    return [b for b in blocks if len(b) >= 2]


def _positions(seq: np.ndarray, n: int) -> dict[tuple[int, int], int]:
    """(job, op_index) -> its position in the sequence (the k-th occurrence of that job)."""
    count = np.zeros(n, dtype=np.int64)
    out = {}
    for pos, job in enumerate(seq):
        job = int(job)
        out[(job, int(count[job]))] = pos
        count[job] += 1
    return out


def n5_local_search(x: np.ndarray, inst, *, max_passes: int = 1000):
    """
    N5 / Nowicki-Smutnicki: swap the first two and the last two operations of every critical
    block. First-improvement, repeated until no move improves.

    Returns (x', cmax', evaluations, improving_moves). `evaluations` counts candidate solutions
    scored and is reported SEPARATELY -- it is NEVER added to `fes` (D3, U7).

    Two ops adjacent on a machine always belong to different jobs, so exchanging their sequence
    positions cannot disturb any job's internal operation order.
    """
    route, ptime = inst.arrays()
    seq = rk_to_job_sequence(x, inst.n).astype(np.int32)
    best = int(evaluate_fast(rk_from_sequence(seq, inst.n), route, ptime, inst.n, inst.m))

    evaluations = 0
    moves = 0
    for _ in range(max_passes):
        improved = False
        for block in _critical_blocks(seq, inst):
            pos = _positions(seq, inst.n)
            candidates = [(block[0], block[1])]
            if len(block) >= 3:
                candidates.append((block[-2], block[-1]))
            for u, v in candidates:
                pu, pv = pos[u], pos[v]
                trial = seq.copy()
                trial[pu], trial[pv] = trial[pv], trial[pu]
                cand_x = rk_from_sequence(trial, inst.n)
                val = int(evaluate_fast(cand_x, route, ptime, inst.n, inst.m))
                evaluations += 1
                if val < best:
                    seq, best = trial, val
                    moves += 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return rk_from_sequence(seq, inst.n), best, evaluations, moves


# ==============================================================================================
# D4 -- the runner
# ==============================================================================================
@dataclass
class JSSPConfig:
    """DTSA's algorithm parameters + ATSA's experimental protocol. See the DTSA adaptation notes D4."""
    n_trees: int | None = None       # None => N = D (DTSA-literal). 40 => ATSA-matched.
    NS: int = 6                      # DTSA §5.2
    ST: float = 0.5                  # DTSA §5.2
    fe_multiplier: int = 1000        # Sahman Table 4: MaxFEs = D * 1000   [DEVIATION from DTSA]
    symmetry_sampler: str = "C1"     # U1; frozen by Gate 1 before any job-shop run
    use_local_search: bool = False   # False => DTSA-core (the comparison column); True => DTSA+LS
    seed_with_mwkr: bool = True      # D2

    def trees_for(self, D: int) -> int:
        return D if self.n_trees is None else self.n_trees

    def n_setting_slug(self) -> str:
        return "N_eq_D" if self.n_trees is None else f"N{self.n_trees}"


def run_one(instance: str, seed: int, jcfg: JSSPConfig) -> dict:
    """One seeded run. Returns a CSV row."""
    inst = load_ta(instance)
    D = inst.D
    N = jcfg.trees_for(D)
    max_fes = D * jcfg.fe_multiplier
    route, ptime = inst.arrays()

    def evaluate(x: np.ndarray) -> float:
        return float(evaluate_fast(x, route, ptime, inst.n, inst.m))

    rng = np.random.default_rng(seed)
    pop = rng.uniform(RK_LO, RK_HI, (N, D))            # exactly ATSA's initialisation
    if jcfg.seed_with_mwkr:                            # D2: Fig. 6 line 5's analogue
        pop[0] = rk_from_sequence(mwkr_sequence(inst), inst.n)

    cfg = Config(N=N, max_fes=max_fes, ST=jcfg.ST, NS=jcfg.NS, seed=seed,
                 symmetry_sampler=jcfg.symmetry_sampler, two_opt_enabled=False)

    t0 = time.perf_counter()
    res = dtsa(evaluate, pop, cfg)
    check_fe_accounting(res, cfg)                      # fes == N + 6*N*iterations
    ls_evals = ls_moves = 0
    cmax = int(res.best_pre_2opt)
    if jcfg.use_local_search:                          # D3: counted separately, never in fes
        _, cmax, ls_evals, ls_moves = n5_local_search(res.best_vector, inst)
    wall = time.perf_counter() - t0

    return {
        # --- columns shared with src/atsa_jssp/experiment.py so the CSVs concatenate ----------
        "instance": instance, "n": inst.n, "m": inst.m, "D": D,
        "algorithm": "DTSA", "seed": seed, "cmax": cmax,
        "fes_used": res.fes, "max_fes": max_fes, "iters": res.iterations,
        "wall_s": round(wall, 3),
        "st_sense": "", "operator_space": "", "branch_granularity": "", "strict_fe_cap": "",
        "N": N, "ST": jcfg.ST, "L": jcfg.NS, "U": jcfg.NS, "dmin": RK_LO, "dmax": RK_HI,
        "git_sha": _git_sha(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fe_per_seed": 1.0,                            # DTSA: exactly 1 FE per seed, always
        "n_seeds": res.fes - N,
        "branch_A_swap": "", "branch_B_sym_shift": "", "branch_C_eq3": "",
        "branch_D_eq4": "", "branch_E_eq_perdim": "",
        # --- DTSA-specific -------------------------------------------------------------------
        "config": "DTSA+LS" if jcfg.use_local_search else "DTSA-core",
        "n_setting": "N=D" if jcfg.n_trees is None else f"N={jcfg.n_trees}",
        "sampler": jcfg.symmetry_sampler,
        "pre_local_search_cmax": int(res.best_pre_2opt),
        "local_search_evals": ls_evals,
        "local_search_moves": ls_moves,
        "seeded_with_mwkr": jcfg.seed_with_mwkr,
    }


CSV_COLUMNS = [
    "instance", "n", "m", "D", "algorithm", "seed", "cmax", "fes_used", "max_fes",
    "iters", "wall_s", "st_sense", "operator_space", "branch_granularity",
    "strict_fe_cap", "N", "ST", "L", "U", "dmin", "dmax", "git_sha", "timestamp",
    "fe_per_seed", "n_seeds", "branch_A_swap", "branch_B_sym_shift",
    "branch_C_eq3", "branch_D_eq4", "branch_E_eq_perdim",
    "config", "n_setting", "sampler", "pre_local_search_cmax",
    "local_search_evals", "local_search_moves", "seeded_with_mwkr",
]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=HERE.parent).stdout.strip()
    except Exception:                                  # noqa: BLE001
        return ""


def system_config() -> dict:
    """
    Exact machine configuration, written alongside every result set.

    The lab meeting requires this with every result, and it is also the only way a later reader
    can tell whether a wall-clock number is comparable. Memory is read through the Windows
    commit API because psutil is not installed (the project design notes §10).
    """
    total_gb = avail_gb = commit_gb = None
    try:
        import ctypes

        class _M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _M()
        m.dwLength = ctypes.sizeof(_M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        total_gb = round(m.ullTotalPhys / 1024 ** 3, 2)
        avail_gb = round(m.ullAvailPhys / 1024 ** 3, 2)
        commit_gb = round(m.ullTotalPageFile / 1024 ** 3, 2)
    except Exception:                                  # noqa: BLE001
        pass

    try:
        import numba
        numba_version = numba.__version__
    except Exception:                                  # noqa: BLE001
        numba_version = None

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git_sha(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "ram_total_gb": total_gb,
        "ram_available_gb": avail_gb,
        "commit_limit_gb": commit_gb,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": numba_version,
    }


def run_experiment(instances: list[str], jcfg: JSSPConfig, *, runs: int = 20,
                   out: pathlib.Path | None = None, jobs: int = 8) -> pathlib.Path:
    """
    D4's protocol: 20 runs, seeds 0..runs-1, one CSV. Checkpointed per instance and resumable,
    in the style of run_tsa_bisect.py -- long runs have been lost to teardown three times
    (the project design notes §9 item 9).

    NEVER pass jobs=-1 (the project design notes §10).
    """
    if jobs < 1:
        raise SystemExit("jobs must be >= 1. -1 has crashed this machine three times.")
    from joblib import Parallel, delayed

    out = out or HERE / "results" / f"dtsa_jssp_{jcfg.n_setting_slug()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{out.stem}_system_config.json").write_text(
        json.dumps(system_config(), indent=2), encoding="utf-8")

    done: set[str] = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            counts: dict[str, int] = {}
            for row in csv.DictReader(f):
                counts[row["instance"]] = counts.get(row["instance"], 0) + 1
        done = {k for k, v in counts.items() if v >= runs}

    for name in instances:
        if name in done:
            continue
        rows = Parallel(n_jobs=jobs, backend="loky")(
            delayed(run_one)(name, s, jcfg) for s in range(runs))
        new = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerows(rows)
        cm = [r["cmax"] for r in rows]
        print(f"{name:<6} mean {sum(cm) / len(cm):9.1f}  min {min(cm):>6}  max {max(cm):>6}")
    return out


if __name__ == "__main__":
    raise SystemExit(
        "dtsa_jssp.py is not runnable as a script yet.\n"
        "The 40-instance run is BLOCKED until Gate 1 passes (the DTSA adaptation notes D5).\n"
        "For the permitted 2-seed ta01 smoke check, use dtsa/smoke_test_jssp.py."
    )
