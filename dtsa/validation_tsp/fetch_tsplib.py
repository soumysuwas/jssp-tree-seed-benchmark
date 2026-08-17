#!/usr/bin/env python3
"""
Fetch the TSPLIB instances needed for the DTSA validation gates into data/tsplib/.

BOUNDARY: writes ONLY under data/tsplib/. It never touches data/raw/ or data/bounds/ — those are
the Taillard originals (the project data-integrity rule). There is no --force flag here and no code path that
writes outside dtsa/.

The canonical TSPLIB host (comopt.ifi.uni-heidelberg.de) is unreachable from this machine, the
same way Taillard's own host is (see data/README.md). So we fetch from mirrors and defend
against a bad mirror two ways:

  1. TWO INDEPENDENT MIRRORS per file where available; the bytes must match exactly.
  2. SELF-VALIDATION downstream: dtsa/validation_tsp/verify_berlin52.py recomputes the optimal tour's length
     from the fetched coordinates and checks it against the published optimum. A mirror with
     corrupted coordinates cannot survive that.

Usage:  uv run python dtsa/validation_tsp/fetch_tsplib.py
"""
from __future__ import annotations

import hashlib
import pathlib
import ssl
import urllib.request
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "data" / "tsplib"

MIRRORS = {
    "mastqe": "https://raw.githubusercontent.com/mastqe/tsplib/master",
    "jorlib": ("https://raw.githubusercontent.com/coin-or/jorlib/master/"
               "jorlib-core/src/test/resources/tspLib/tsp"),
    "tsplibnet": "https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp",
}

# file -> mirrors to try, in order. First success is authoritative; every other success must
# match it byte for byte.
WANTED = {
    "berlin52.tsp":      ["mastqe", "jorlib", "tsplibnet"],
    "berlin52.opt.tour": ["tsplibnet", "jorlib"],
    "kroA100.tsp":       ["mastqe", "jorlib", "tsplibnet"],
    "kroA100.opt.tour":  ["tsplibnet", "jorlib"],
    "kroB100.tsp":       ["mastqe", "jorlib", "tsplibnet"],
    "kroC100.tsp":       ["mastqe", "jorlib", "tsplibnet"],
    "kroD100.tsp":       ["mastqe", "jorlib", "tsplibnet"],
    "kroE100.tsp":       ["mastqe", "jorlib", "tsplibnet"],
}

CTX = ssl.create_default_context()


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30, context=CTX) as r:
        return r.read()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record: list[dict] = []

    for name, mirror_names in WANTED.items():
        blobs: dict[str, bytes] = {}
        for mn in mirror_names:
            url = f"{MIRRORS[mn]}/{name}"
            try:
                blobs[mn] = get(url)
                print(f"  ok    {mn:<10} {len(blobs[mn]):>6} B  {name}")
            except Exception as e:  # noqa: BLE001 - a dead mirror is expected, not exceptional
                print(f"  miss  {mn:<10} {type(e).__name__}: {str(e)[:50]}  {name}")

        if not blobs:
            raise SystemExit(f"FATAL: no mirror served {name}")

        authoritative_mirror = mirror_names[0] if mirror_names[0] in blobs else next(iter(blobs))
        data = blobs[authoritative_mirror]

        # Cross-check: every other mirror that answered must agree byte for byte.
        agree, disagree = [], []
        for mn, b in blobs.items():
            if mn == authoritative_mirror:
                continue
            (agree if b == data else disagree).append(mn)
        if disagree:
            raise SystemExit(
                f"FATAL: mirrors disagree on {name}: {authoritative_mirror} vs {disagree}. "
                "Do not guess which is right — resolve by hand."
            )

        (OUT / name).write_bytes(data)
        record.append({
            "name": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "url": f"{MIRRORS[authoritative_mirror]}/{name}",
            "mirror": authoritative_mirror,
            "confirmed_by": agree,
        })
        print(f"  WROTE {name}  ({len(data)} B, confirmed by {len(agree)} other mirror(s))\n")

    write_provenance(record)


def write_provenance(record: list[dict]) -> None:
    """Mirrors the shape of what scripts/write_provenance.py emits for data/."""
    lines = [
        "# DTSA Data Provenance",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        "`dtsa/validation_tsp/fetch_tsplib.py`.",
        "",
        "Reinelt, G. (1991). *TSPLIB — A Traveling Salesman Problem Library.* ORSA Journal on",
        "Computing 3(4):376-384. doi:10.1287/ijoc.3.4.376",
        "",
        "## Scope",
        "",
        "These files exist only to run the DTSA **TSP validation gates** "
        "(`the DTSA adaptation notes` D5). They are unrelated to the job-shop data and live in a "
        "separate tree on purpose: **`data/raw/` and `data/bounds/` are the Taillard originals and "
        "are never written by anything in `dtsa/`.**",
        "",
        "## Canonical origin (for citation, not download)",
        "",
        "`http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/` — the authoritative TSPLIB "
        "host. **Unreachable from this machine** (connection timeout), exactly as Taillard's own "
        "host is (see `data/README.md`). Files were therefore taken from the mirrors below "
        "and cross-checked against each other byte for byte.",
        "",
        "## SHA256 and exact source URL",
        "",
        "| file | bytes | sha256 | fetched from | byte-identical on |",
        "|---|---|---|---|---|",
    ]
    for r in record:
        confirmed = ", ".join(r["confirmed_by"]) if r["confirmed_by"] else "_(single mirror)_"
        lines.append(
            f"| `data/tsplib/{r['name']}` | {r['bytes']} | `{r['sha256']}` | "
            f"`{r['url']}` | {confirmed} |"
        )
    lines += [
        "",
        "## Mirrors used",
        "",
        "| key | base URL |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | `{v}` |" for k, v in MIRRORS.items()]
    lines += [
        "",
        "## Validation",
        "",
        "Byte-agreement between independent mirrors proves only that they share an ancestor. The "
        "load-bearing check is **`dtsa/validation_tsp/verify_berlin52.py`**, which recomputes the length of the "
        "published optimal tour from the fetched coordinates under both TSPLIB distance "
        "conventions and compares it with the optima printed in Cinar et al. (2020) §5. A mirror "
        "with corrupted coordinates cannot pass it. Its result is asserted on every test run by "
        "`dtsa/tests/test_berlin52.py`.",
        "",
    ]
    # Print provenance to stdout only; the committed data provenance is data/README.md.
    print("\n".join(lines))
    print(f"\n# {len(record)} files hashed. (Printed to stdout; committed provenance is data/README.md.)")


if __name__ == "__main__":
    main()
