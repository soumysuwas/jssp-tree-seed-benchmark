"""
Instance parsers. Ported from reference/instance_reference.py. Two formats matter:

1. TAILLARD ORIGINAL (the supplied Taillard data, what Taillard's own site serves).
   Files: tai15_15.txt ... tai100_20.txt, 10 instances per file.
   Machines are 1-INDEXED. Times block and Machines block are SEPARATE.

       Nb of jobs, Nb of Machines, Time seed, Machine seed, Upper bound, Lower bound
            15        15  840612802  398197754      1231      1005
       Times
        94 66 10 ...        <- row per job: durations in ROUTE ORDER
       Machines
         7 13  5 ...        <- row per job: machine visited at each step (1-indexed)

   Read the two blocks POSITIONALLY: job 1 spends 94 on machine 7, then 66 on
   machine 13, then 10 on machine 5, ...  The route is given, not generated.

2. JSPLIB / OR-Library (github.com/tamy0612/JSPLIB).
   Machines are 0-INDEXED, interleaved (machine time) pairs, one line per job.

       15 15
        6 94 12 66  4 10 ...

Both are supported and are provably the same data -- see the design notes.

The header's Upper/Lower bound columns are STALE (the design notes §7) -- they are parsed
for provenance only. Never use them as the gap-to-optimum reference; use data/bounds/.
`Time seed` / `Machine seed` are Taillard's 1993 instance-GENERATION seeds. They are not
algorithm inputs and must never seed ATSA.
"""
from __future__ import annotations
import pathlib
import numpy as np

from atsa_jssp.decoder import Instance


def load_taillard_original(path) -> list[Instance]:
    """Parse one tai{n}_{m}.txt file -> list of 10 Instances, in file order."""
    lines = [l.strip() for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]
    out, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("Nb of jobs"):
            i += 1
            continue
        n, m, tseed, mseed, ub, lb = map(int, lines[i + 1].split())
        assert lines[i + 2] == "Times", f"expected 'Times' at line {i+2}"
        ptime = np.array([list(map(int, lines[i + 3 + r].split()))
                          for r in range(n)], dtype=np.int32)
        assert lines[i + 3 + n] == "Machines", f"expected 'Machines' at line {i+3+n}"
        route = np.array([list(map(int, lines[i + 4 + n + r].split()))
                          for r in range(n)], dtype=np.int32) - 1     # -> 0-indexed
        out.append(Instance(f"{path}#{len(out)}", n, m, route, ptime, ub=ub, lb=lb))
        i += 4 + 2 * n
    return out


def load_jsplib(path) -> Instance:
    """Parse a JSPLIB instance file (0-indexed, interleaved pairs)."""
    raw = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    lines = [l for l in raw if l.strip() and not l.lstrip().startswith("#")]
    n, m = map(int, lines[0].split())
    route = np.zeros((n, m), dtype=np.int32)
    ptime = np.zeros((n, m), dtype=np.int32)
    for r in range(n):
        v = list(map(int, lines[1 + r].split()))
        route[r] = v[0::2]
        ptime[r] = v[1::2]
    return Instance(pathlib.Path(path).stem, n, m, route, ptime)


# Which file holds which ta*, and which 5 the paper uses.
TAILLARD_FILES = {
    "tai15_15.txt":  (1, 10),    # ta01-ta10   15x15   D=225   paper uses ta01-05
    "tai20_15.txt":  (11, 20),   # ta11-ta20   20x15   D=300   paper uses ta11-15
    "tai20_20.txt":  (21, 30),   # ta21-ta30   20x20   D=400   paper uses ta21-25
    "tai30_15.txt":  (31, 40),   # ta31-ta40   30x15   D=450   paper uses ta31-35
    "tai30_20.txt":  (41, 50),   # ta41-ta50   30x20   D=600   paper uses ta41-45
    "tai50_15.txt":  (51, 60),   # ta51-ta60   50x15   D=750   paper uses ta51-55
    "tai50_20.txt":  (61, 70),   # ta61-ta70   50x20   D=1000  paper uses ta61-65
    "tai100_20.txt": (71, 80),   # ta71-ta80   100x20  D=2000  paper uses ta71-75
}
PAPER_INSTANCES = [f"ta{k:02d}" for base in (1, 11, 21, 31, 41, 51, 61, 71)
                   for k in range(base, base + 5)]


def load_ta(name: str, data_dir="data/raw") -> Instance:
    """load_ta('ta23') -> Instance. Works off the Taillard-original files."""
    num = int(name[2:])
    for fname, (lo, hi) in TAILLARD_FILES.items():
        if lo <= num <= hi:
            inst = load_taillard_original(pathlib.Path(data_dir) / fname)[num - lo]
            inst.name = name
            return inst
    raise KeyError(f"{name} is not in ta01..ta80")
