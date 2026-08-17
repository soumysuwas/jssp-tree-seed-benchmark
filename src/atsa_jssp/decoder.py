"""
RK decoder + schedule builder.

Ported from reference/decoder_reference.py, which is VERIFIED against Sahman (2022)
Table 2 and Figure 1. The NumPy path here is a line-for-line port and stays as the
ORACLE -- never delete it, never "improve" it.

`evaluate_fast` is the numba @njit(cache=True) twin of `evaluate`. It must agree with
the oracle on every input; tests/test_golden.py::test_njit_matches_oracle enforces this
on 30,000 random vectors. The njit path is what ATSA actually calls: it runs D*1000
times per run and is ~68x the NumPy path (the design notes §2).

Both paths produce a SEMI-ACTIVE schedule. No left-shifting -- the paper has none and
adding it would break the reproduction (the design notes §4).
"""
from __future__ import annotations
import numpy as np
from atsa_jssp._compat import njit  # optional-Numba shim (pure-Python fallback)


# ----------------------------------------------------------------------------
# Instance container
# ----------------------------------------------------------------------------
class Instance:
    """
    n     : number of jobs
    m     : number of machines
    route : (n, m) int array. route[i, j] = machine (0-indexed) of the j-th
            operation of job i.  Operation order within a job is FIXED.
    ptime : (n, m) int array. ptime[i, j] = processing time of the j-th
            operation of job i (on machine route[i, j]).
    """

    def __init__(self, name, n, m, route, ptime, ub=None, lb=None):
        self.name, self.n, self.m = name, n, m
        self.route = np.asarray(route, dtype=np.int32)
        self.ptime = np.asarray(ptime, dtype=np.int32)
        self.ub, self.lb = ub, lb
        assert self.route.shape == (n, m) and self.ptime.shape == (n, m)
        # every job must visit every machine exactly once
        for i in range(n):
            assert sorted(self.route[i].tolist()) == list(range(m)), \
                f"job {i} route is not a permutation of machines"

    @property
    def D(self) -> int:
        return self.n * self.m

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(route, ptime) as int64 -- the layout evaluate_fast wants."""
        return self.route.astype(np.int64), self.ptime.astype(np.int64)


# ----------------------------------------------------------------------------
# Bean random-key encoding -- the paper's Eq. 2
# ----------------------------------------------------------------------------
def rk_to_job_sequence(x: np.ndarray, n: int) -> np.ndarray:
    """
    Eq.2:  S = (tau_l mod n) + 1,  l = 1..n*m

    tau = rank of x[l] when x is sorted ascending (1-based dense ranks).
    Returns a 0-INDEXED job sequence of length n*m where each job id appears
    exactly m times.

    Why it is always feasible: tau is a permutation of 1..n*m, and the residues
    of 1..n*m mod n hit each class exactly m times. No repair is ever needed.

    Tie-breaking: ties in x are broken by position (stable argsort). With
    continuous randoms ties are measure-zero, but this makes it deterministic.
    """
    order = np.argsort(x, kind="stable")          # positions sorted by value
    tau = np.empty(x.shape[0], dtype=np.int64)
    tau[order] = np.arange(1, x.shape[0] + 1)     # 1-based ranks
    return (tau % n).astype(np.int32)             # paper's "+1" is 1-indexing; we keep 0-indexed


def rk_ranks(x: np.ndarray) -> np.ndarray:
    """The paper's 'Integer Series (phi)' row, 1-based. Exposed for tests."""
    order = np.argsort(x, kind="stable")
    tau = np.empty(x.shape[0], dtype=np.int64)
    tau[order] = np.arange(1, x.shape[0] + 1)
    return tau


# ----------------------------------------------------------------------------
# Schedule builder (semi-active, left-shift-free) -- the oracle
# ----------------------------------------------------------------------------
def build_schedule(seq: np.ndarray, inst: Instance):
    """
    Decode a job sequence (0-indexed job ids, each appearing m times) into a
    schedule and return (Cmax, starts, ends).

    Rule: process the sequence left to right. The k-th occurrence of job i is
    its k-th operation. Schedule it at the earliest time >= max(machine free,
    job ready). This enforces BOTH constraints structurally:
      - a machine handles one operation at a time  (machine_free)
      - a job follows its fixed route in order     (job_ready + op counter)
    so no constraint checking or repair is needed anywhere.
    """
    n, m = inst.n, inst.m
    machine_free = np.zeros(m, dtype=np.int64)
    job_ready = np.zeros(n, dtype=np.int64)
    op_idx = np.zeros(n, dtype=np.int32)
    starts = np.zeros((n, m), dtype=np.int64)
    ends = np.zeros((n, m), dtype=np.int64)

    for job in seq:
        j = op_idx[job]
        mach = inst.route[job, j]
        dur = inst.ptime[job, j]
        st = machine_free[mach] if machine_free[mach] > job_ready[job] else job_ready[job]
        en = st + dur
        starts[job, j] = st
        ends[job, j] = en
        machine_free[mach] = en
        job_ready[job] = en
        op_idx[job] += 1

    return int(job_ready.max()), starts, ends


def evaluate(x: np.ndarray, inst: Instance) -> int:
    """The objective function (ORACLE). One call == one FE."""
    seq = rk_to_job_sequence(x, inst.n)
    cmax, _, _ = build_schedule(seq, inst)
    return cmax


# ----------------------------------------------------------------------------
# numba twin -- this is the one ATSA calls
# ----------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def evaluate_fast(x, route, ptime, n, m):
    """
    njit twin of `evaluate`. Fused: ranks -> Eq.2 -> schedule build -> Cmax, with no
    intermediate allocation beyond the three counters.

    `kind="mergesort"` is numba's stable sort, matching the oracle's
    np.argsort(kind="stable"). Ties are measure-zero on continuous randoms, but the
    operators permute values rather than regenerating them, so a stable rule keeps the
    two paths bit-identical regardless.

    route/ptime must be (n, m) int64 -- use Instance.arrays().
    """
    D = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    tau = np.empty(D, dtype=np.int64)
    for k in range(D):
        tau[order[k]] = k + 1                      # 1-based ranks

    machine_free = np.zeros(m, dtype=np.int64)
    job_ready = np.zeros(n, dtype=np.int64)
    op_idx = np.zeros(n, dtype=np.int64)

    for l in range(D):
        job = tau[l] % n                           # Eq.2, 0-indexed
        j = op_idx[job]
        mach = route[job, j]
        st = machine_free[mach]
        if job_ready[job] > st:
            st = job_ready[job]
        en = st + ptime[job, j]
        machine_free[mach] = en
        job_ready[job] = en
        op_idx[job] += 1

    cmax = 0
    for i in range(n):
        if job_ready[i] > cmax:
            cmax = job_ready[i]
    return cmax


# ----------------------------------------------------------------------------
# The paper's 2x3 worked example (Table 1 / Table 2 / Figure 1)
# ----------------------------------------------------------------------------
def paper_2x3_instance() -> Instance:
    """
    Table 1:
        Job | Machine1 Machine2 Machine3 | Processing Order
         1  |    8         6        3    |   {1,3,2}
         2  |    4        10        8    |   {2,1,3}
    NOTE the times are given PER MACHINE (columns), while the order is a
    separate column. So job 1 runs: M1 for 8, then M3 for 3, then M2 for 6.
    This re-indexing is the #1 place people get the example wrong.
    """
    order = np.array([[1, 3, 2], [2, 1, 3]]) - 1              # -> 0-indexed machines
    per_machine_time = np.array([[8, 6, 3], [4, 10, 8]])      # indexed by machine
    ptime = np.array([[per_machine_time[i, order[i, j]] for j in range(3)]
                      for i in range(2)])
    return Instance("paper2x3", 2, 3, order, ptime)
