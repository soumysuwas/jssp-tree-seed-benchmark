#!/usr/bin/env python3
"""
Prove the Taillard-original files and the JSPLIB mirror describe the SAME 80 instances.

This is the check that lets us say to reviewers: "the supplied data is
the standard Taillard benchmark, verified operation-by-operation, not merely assumed."

Usage:  uv run python scripts/verify_data.py
"""
from __future__ import annotations
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "reference"))
import numpy as np
from instance_reference import load_taillard_original, load_jsplib, TAILLARD_FILES

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main() -> int:
    ok = bad = missing = 0
    for fname, (lo, hi) in TAILLARD_FILES.items():
        p = ROOT / "data/raw" / fname
        if not p.exists():
            print(f"  -- {fname}: MISSING (run scripts/fetch_data.py)")
            missing += hi - lo + 1
            continue
        insts = load_taillard_original(p)
        assert len(insts) == 10, f"{fname}: expected 10 instances, got {len(insts)}"
        for k, a in enumerate(insts):
            name = f"ta{lo + k:02d}"
            q = ROOT / "data/jsplib" / name
            if not q.exists():
                missing += 1
                continue
            b = load_jsplib(q)
            same = (a.n == b.n and a.m == b.m
                    and np.array_equal(a.route, b.route)
                    and np.array_equal(a.ptime, b.ptime))
            if same:
                ok += 1
            else:
                bad += 1
                print(f"  !! {name}: MISMATCH between Taillard-original and JSPLIB")

    if ok == 0 and missing == 80 and (ROOT / "data/raw/tai15_15.txt").exists():
        print("\nJSPLIB cross-check copies not present (data/jsplib/ is empty).")
        print("Run `uv run python scripts/fetch_data.py` once to enable the cross-check.")
        print("This is a nice-to-have proof, NOT a dependency -- do not let it block you.")
        return 2

    print(f"\n{ok}/80 instances agree across formats"
          f"{f'  ({bad} mismatched)' if bad else ''}"
          f"{f'  ({missing} missing)' if missing else ''}")
    if bad:
        print("A mismatch means the PARSER is wrong (both sources are canonical).")
        print("Check: machines are 1-indexed in Taillard-original, 0-indexed in JSPLIB.")
        return 1
    if missing:
        return 2
    print("VERIFIED: the supplied data == Taillard's benchmark == JSPLIB, exactly.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
