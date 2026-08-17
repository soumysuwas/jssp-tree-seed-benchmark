"""
20-run experiment harness. CSV schema is the design notes §3 — one row per
run, carrying its FULL config and seed, so any number can be traced back to what produced
it six weeks later.

Windows notes (the design notes §5):
  - NUMBA_NUM_THREADS / OMP_NUM_THREADS are set to 1 HERE, before numba is imported, or
    16 loky workers each spawn 16 numba threads -> 256 threads on 16 cores -> slower than
    serial. This is the #1 performance bug in this kind of harness.
  - every entry point that spawns workers must be guarded by `if __name__ == "__main__":`
    or Windows spawn fork-bombs.
"""
from __future__ import annotations
import os

# MUST precede any numba import, including the transitive one via .atsa
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pathlib
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
from joblib import Parallel, delayed

from atsa_jssp.atsa import Config, atsa, check_fe_accounting
from atsa_jssp.instance import PAPER_INSTANCES, load_ta

ROOT = pathlib.Path(__file__).resolve().parents[2]

CSV_COLUMNS = [
    "instance", "n", "m", "D", "algorithm", "seed", "cmax", "fes_used", "max_fes",
    "iters", "wall_s", "st_sense", "operator_space", "branch_granularity",
    "strict_fe_cap", "N", "ST", "L", "U", "dmin", "dmax", "git_sha", "timestamp",
    # diagnostics beyond the spec'd schema — cheap, and they are what prove the FE
    # accounting was right for THIS row rather than in general
    "fe_per_seed", "n_seeds", "branch_A_swap", "branch_B_sym_shift",
    "branch_C_eq3", "branch_D_eq4", "branch_E_eq_perdim",
]


def _commit_available_gb() -> float | None:
    """Windows: RAM + pagefile still committable. None elsewhere."""
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
        return m.ullAvailPageFile / 1024 ** 3
    except Exception:                                   # noqa: BLE001 — advisory only
        return None


# Each loky worker is a whole Python process carrying its own numba/LLVM runtime — measured
# at roughly this much resident, and it dwarfs the algorithm's own arrays (a D=2000 population
# is only ~0.6 MB). This is what actually sizes the worker budget.
WORKER_GB = 0.9
RESERVE_GB = 6.0                                        # leave the desktop usable


def safe_n_jobs(requested: int) -> int:
    """
    Clamp worker count to what memory can actually commit.

    Why this exists: this box has 31.8 GB RAM but a ~33.8 GB commit limit (almost no pagefile).
    24 numba workers x ~0.9 GB on top of a normal desktop exceeds it, and Windows responds by
    killing workers (joblib TerminatedWorkerError) or destabilising the machine. That happened
    three times during the TSA bisection, once taking the whole system down. `-1` is not a safe
    default for numba workloads on this machine.
    """
    n = os.cpu_count() or 1
    want = n if requested in (-1, 0) else min(requested, n)
    avail = _commit_available_gb()
    if avail is None:
        return want
    budget = max(1, int((avail - RESERVE_GB) // WORKER_GB))
    capped = max(1, min(want, budget))
    if capped < want:
        print(f"[experiment] capping workers {want} -> {capped} "
              f"({avail:.1f} GB committable, ~{WORKER_GB} GB/worker, {RESERVE_GB} GB reserved)")
    return capped


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:                                   # noqa: BLE001 — provenance is best-effort
        return "unknown"


def run_one(instance: str, seed: int, cfg: Config, sha: str = "",
            algorithm: str = "ATSA") -> dict:
    """One seeded run -> one CSV row. Pure; safe to call in a worker process."""
    inst = load_ta(instance, ROOT / "data/raw")
    t0 = time.perf_counter()
    if algorithm.upper() == "TSA":
        from atsa_jssp.tsa import tsa
        res = tsa(inst, seed, cfg)
    else:
        res = atsa(inst, seed, cfg)
    wall = time.perf_counter() - t0

    acc = check_fe_accounting(res, cfg, algorithm=algorithm)   # raises if a branch is mis-wired
    L, U = cfg.limits()
    bc = res["branch_counts"]
    return {
        "instance": instance, "n": inst.n, "m": inst.m, "D": inst.D,
        "algorithm": algorithm.upper(), "seed": seed, "cmax": res["cmax"],
        "fes_used": res["fes"], "max_fes": inst.D * cfg.fe_multiplier,
        "iters": res["iters"], "wall_s": round(wall, 3),
        "st_sense": cfg.st_sense, "operator_space": cfg.operator_space,
        "branch_granularity": cfg.branch_granularity, "strict_fe_cap": cfg.strict_fe_cap,
        "N": cfg.N, "ST": cfg.ST, "L": L, "U": U, "dmin": cfg.dmin, "dmax": cfg.dmax,
        "git_sha": sha or git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fe_per_seed": round(acc["fe_per_seed"], 4), "n_seeds": res["n_seeds"],
        "branch_A_swap": bc["A_swap"], "branch_B_sym_shift": bc["B_sym_shift"],
        "branch_C_eq3": bc["C_eq3"], "branch_D_eq4": bc["D_eq4"],
        "branch_E_eq_perdim": bc["E_eq_perdim"],
    }


def run_experiment(instances: list[str], runs: int = 20, cfg: Config = Config(),
                   jobs: int = -1, out: pathlib.Path | None = None,
                   algorithm: str = "ATSA") -> pd.DataFrame:
    """
    `runs` independent seeded runs (seeds 0..runs-1) of every instance, in parallel.

    Instances are ordered by D so the cheap ones bank first — the design notes §4: ta71-75 is 73%
    of the total compute, so it must come last or a deadline overrun costs you everything.
    """
    order = {name: i for i, name in enumerate(PAPER_INSTANCES)}
    instances = sorted(instances, key=lambda s: (load_ta(s, ROOT / "data/raw").D,
                                                 order.get(s, 999)))
    sha = git_sha()
    tasks = [(name, seed) for name in instances for seed in range(runs)]
    rows = Parallel(n_jobs=safe_n_jobs(jobs), backend="loky", verbose=5)(
        delayed(run_one)(name, seed, cfg, sha, algorithm) for name, seed in tasks
    )
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-instance mean/med/min/max/std — the shape of the paper's Table 5."""
    g = df.groupby("instance")["cmax"]
    out = pd.DataFrame({
        "runs": g.count(), "mean": g.mean().round(2), "med": g.median(),
        "min": g.min(), "max": g.max(), "std": g.std().round(2),
    })
    out["wall_s_mean"] = df.groupby("instance")["wall_s"].mean().round(2)
    out["fes_min"] = df.groupby("instance")["fes_used"].min()
    out["fes_max"] = df.groupby("instance")["fes_used"].max()
    return out.reset_index()


if __name__ == "__main__":                              # REQUIRED on Windows
    df = run_experiment(["ta01"], runs=20, out=ROOT / "results/ta01_gate4.csv")
    print(summarise(df).to_string(index=False))
