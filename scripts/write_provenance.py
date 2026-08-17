#!/usr/bin/env python3
"""
Print SHA256 of every instance file we actually used (to stdout; writes no file).
The authoritative, committed provenance document is data/README.md.

scripts/fetch_data.py has a write_provenance() but it only records files IT fetched, and we
deliberately never let it touch data/raw/ (the Taillard originals must not be overwritten). So the
8 Taillard files it did not download had no SHA recorded anywhere. This hashes what is on disk.

Usage:  uv run python scripts/write_provenance.py
"""
from __future__ import annotations

import hashlib
import pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]

TAILLARD_URL = ("http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/"
                "jobshop.dir")
JSPLIB_URL = "https://raw.githubusercontent.com/tamy0612/JSPLIB/master"


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    lines = [
        "# Data Provenance",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        "`scripts/write_provenance.py`.",
        "",
        "Taillard, E. (1993). *Benchmarks for basic scheduling problems.* European Journal of",
        "Operational Research 64(2):278-285. doi:10.1016/0377-2217(93)90182-M",
        "",
        "## Source of record",
        "",
        "`data/raw/` and `data/bounds/best_lb_up.txt` came from **the standard Taillard distribution: "
        "the standard Taillard distribution** and were copied in, never downloaded and "
        "never overwritten. They are the complete Taillard job-shop set: 8 files x 10 "
        "instances = ta01-ta80, plus Taillard's own bounds file.",
        "",
        "`data/jsplib/` holds the JSPLIB mirror copies, fetched once purely to CROSS-CHECK the "
        "above. `tests/test_golden.py::test_all_80_instances_agree` and "
        "`scripts/verify_data.py` prove the two sources describe the same 80 instances "
        "operation-by-operation (80/80, modulo 0- vs 1-indexed machines).",
        "",
        "## SHA256 — the benchmark source originals (authoritative)",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    for p in sorted((ROOT / "data/raw").glob("tai*.txt")):
        lines.append(f"| `data/raw/{p.name}` | `{sha256(p)}` |")
    b = ROOT / "data/bounds/best_lb_up.txt"
    if b.exists():
        lines.append(f"| `data/bounds/best_lb_up.txt` | `{sha256(b)}` |")

    js = sorted((ROOT / "data/jsplib").glob("ta*"))
    if js:
        lines += [
            "",
            "## SHA256 — JSPLIB cross-check copies",
            "",
            f"Fetched from `{JSPLIB_URL}/instances/<name>`. {len(js)} files. Verification only.",
            "",
            "<details><summary>expand</summary>",
            "",
            "| file | sha256 |",
            "|---|---|",
        ]
        lines += [f"| `data/jsplib/{p.name}` | `{sha256(p)}` |" for p in js]
        lines += ["", "</details>"]

    lines += [
        "",
        "## Canonical origin (for citation, not download)",
        "",
        f"Taillard's own host: `{TAILLARD_URL}/`  (tai15_15.txt ... tai100_20.txt, "
        "best_lb_up.txt).",
        "The host is frequently unreachable from India; it is irrelevant to this project "
        "because the files were part of the standard benchmark. Cite it as the benchmark's origin.",
        "",
    ]
    # Print the checksum table to stdout only. The authoritative provenance document is
    # data/README.md; this tool does not write a competing file.
    print("\n".join(lines))
    n = len(list((ROOT / "data/raw").glob("tai*.txt"))) + (1 if b.exists() else 0) + len(js)
    print(f"\n# {n} files hashed. (Printed to stdout; the committed provenance is data/README.md.)")


if __name__ == "__main__":
    main()
