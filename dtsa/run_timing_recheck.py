#!/usr/bin/env python3
"""
D008 B2 -- re-measure the two timing outliers.

In the B1 serial pass ta73 took 249.9 s against 213-220 s for its size peers (ta71/ta72/ta74/ta75,
all D=2000), and ta55 took 32.7 s against 29.3-29.8 s for its peers (ta51-ta54, all D=750). Both
gaps are consistent with background interference rather than with the instances themselves.

Same protocol as B1 -- same seeds, same config, `--jobs 1`, nothing else running -- so the two
passes are directly comparable. Writes a SEPARATE file; the B1 series is not overwritten, and
whichever way this comes out the correction is visible in the diff.

Usage:  uv run python -u dtsa/run_timing_recheck.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from run_timing import SERIAL_SEEDS, append, measure   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "timing" / "timing_recheck.csv"
OUTLIERS = ["ta55", "ta73"]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()          # a re-check is a fresh measurement, never appended to an old one
    for instance in OUTLIERS:
        rows = [measure(instance, s, 1) for s in SERIAL_SEEDS]
        append(OUT, rows)
        mean = sum(r["wall_seconds"] for r in rows) / len(rows)
        runs = ", ".join(f"{r['wall_seconds']:.1f}s" for r in rows)
        print(f"{instance:<6} mean {mean:8.2f}s   ({runs})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
