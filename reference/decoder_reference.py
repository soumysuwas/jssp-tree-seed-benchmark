"""
Reference RK decoder + schedule builder for ATSA/JSSP.

VERIFIED against Şahman (2022) Table 2 and Figure 1. DO NOT CHANGE THE LOGIC.
Port this to numba; keep this file as the oracle to diff against.

Run:  python reference/decoder_reference.py
"""
from __future__ import annotations
import numpy as np


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


# ----------------------------------------------------------------------------
# Bean random-key encoding — the paper's Eq. 2
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
# Schedule builder (semi-active, left-shift-free)
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
    """The objective function. One call == one FE."""
    seq = rk_to_job_sequence(x, inst.n)
    cmax, _, _ = build_schedule(seq, inst)
    return cmax


# ----------------------------------------------------------------------------
# GOLDEN TEST — the paper's own 2x3 example (Table 1 / Table 2 / Figure 1)
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


def test_paper_example():
    inst = paper_2x3_instance()
    assert inst.ptime.tolist() == [[8, 3, 6], [10, 4, 8]], inst.ptime.tolist()

    x = np.array([-2.1, 4.5, 4.9, 1.2, -4.3, 3.8])            # paper's continuous vector
    assert rk_ranks(x).tolist() == [2, 5, 6, 3, 1, 4], "Table 2 'Integer Series' mismatch"

    seq = rk_to_job_sequence(x, 2)
    assert (seq + 1).tolist() == [1, 2, 1, 2, 2, 1], "Table 2 'Job Indexes' mismatch"

    cmax, starts, ends = build_schedule(seq, inst)
    # Figure 1's title says Makespan=22. The body text says 24 -> the text is a typo.
    assert cmax == 22, f"expected 22, got {cmax}"
    # exact Gantt from Figure 1
    assert starts.tolist() == [[0, 8, 11], [0, 10, 14]]
    assert ends.tolist() == [[8, 11, 17], [10, 14, 22]]
    print("PASS  paper 2x3 example: ranks, Eq.2, Gantt and Cmax=22 all reproduce")


def test_feasibility_invariant():
    rng = np.random.default_rng(0)
    for n, m in [(2, 3), (5, 4), (15, 15)]:
        for _ in range(200):
            x = rng.uniform(-5, 5, n * m)
            seq = rk_to_job_sequence(x, n)
            counts = np.bincount(seq, minlength=n)
            assert (counts == m).all(), f"job counts {counts} != m={m}"
    print("PASS  Eq.2 always yields exactly m occurrences per job (no repair needed)")


if __name__ == "__main__":
    test_paper_example()
    test_feasibility_invariant()
