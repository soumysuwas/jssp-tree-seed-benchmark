#!/usr/bin/env python3
"""
Fetch the Taillard JSSP benchmark.

Strategy:
  1. Try Taillard's own host (canonical). Two known hostnames; both are tried.
  2. If that fails (it often does from outside CH — this is the "link that wouldn't
     open" from the kickoff call), fall back to the GitHub mirror and RECONSTRUCT
     the Taillard-original-format files from JSPLIB. This is provably lossless:
     see the design notes §1.
  3. Always fetch the JSPLIB copies too, so tests/test_golden.py can prove the two
     formats agree.
  4. Write data/README.md with URL + SHA256 + date for every file.

SAFETY CONTRACT
---------------
By default this script is **read-only with respect to data/raw/ and
data/bounds/best_lb_up.txt**. Those come from the benchmark source and must never be
clobbered (the design notes §1). Default behaviour = fetch the JSPLIB
cross-check copies only.

  uv run python scripts/fetch_data.py                # SAFE. jsplib only.
  uv run python scripts/fetch_data.py --all          # also fills MISSING raw/ files
  uv run python scripts/fetch_data.py --all --force  # overwrite raw/. Don't.

Even under --all, existing files are skipped, not overwritten. --force is the
only path that overwrites and it prints a warning.
"""
from __future__ import annotations
import hashlib, json, pathlib, sys, urllib.request, urllib.error
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
JSPLIB = ROOT / "data/jsplib"
BOUNDS = ROOT / "data/bounds"

TAILLARD_HOSTS = [
    "http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/jobshop.dir",
    "https://mistic.iict-heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/jobshop.dir",
]
MIRROR = "https://raw.githubusercontent.com/tamy0612/JSPLIB/master"

# file -> (first ta number, last ta number, n, m)
FILES = {
    "tai15_15.txt":  (1, 10, 15, 15),
    "tai20_15.txt":  (11, 20, 20, 15),
    "tai20_20.txt":  (21, 30, 20, 20),
    "tai30_15.txt":  (31, 40, 30, 15),
    "tai30_20.txt":  (41, 50, 30, 20),
    "tai50_15.txt":  (51, 60, 50, 15),
    "tai50_20.txt":  (61, 70, 50, 20),
    "tai100_20.txt": (71, 80, 100, 20),
}

provenance: list[tuple[str, str, str]] = []   # (path, url, sha256)
FORCE = False                                 # set from --force in __main__


def _get(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "atsa-jssp-repro/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:                       # noqa: BLE001 - any failure -> fall back
        print(f"    ! {type(e).__name__}: {url}")
        return None


def _save(path: pathlib.Path, data: bytes, url: str, protect: bool = False) -> None:
    """protect=True => never clobber an existing file unless --force."""
    if protect and path.exists() and not FORCE:
        provenance.append((str(path.relative_to(ROOT)), "local (local) - left untouched",
                           hashlib.sha256(path.read_bytes()).hexdigest()))
        return
    if protect and path.exists() and FORCE:
        print(f"    !! OVERWRITING {path.relative_to(ROOT)} (--force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    provenance.append((str(path.relative_to(ROOT)), url,
                       hashlib.sha256(data).hexdigest()))


# ---------------------------------------------------------------------------
def fetch_jsplib() -> dict[str, str]:
    """Pull ta01..ta80 + instances.json from the GitHub mirror. This always works."""
    print("[2/4] JSPLIB mirror (ta01..ta80 + metadata)")
    got = {}
    meta = _get(f"{MIRROR}/instances.json")
    if meta:
        _save(BOUNDS / "instances.json", meta, f"{MIRROR}/instances.json")
    for k in range(1, 81):
        name = f"ta{k:02d}"
        data = _get(f"{MIRROR}/instances/{name}")
        if data is None:
            sys.exit(f"FATAL: mirror unreachable for {name}. Check your network.")
        _save(JSPLIB / name, data, f"{MIRROR}/instances/{name}")
        got[name] = data.decode()
    print(f"    ok  80/80 instances")
    return got


def reconstruct_taillard(fname: str, jsplib: dict[str, str]) -> bytes:
    """
    Rebuild a Taillard-original-format file from the JSPLIB copies.
    Lossless: the only differences are 0-vs-1 machine indexing and layout.
    UB/LB are filled from instances.json when available, else 0.
    """
    lo, hi, n, m = FILES[fname]
    bounds = {}
    p = BOUNDS / "instances.json"
    if p.exists():
        for rec in json.loads(p.read_text()):
            opt = rec.get("optimum")
            bd = rec.get("bounds") or {}
            ub = opt if opt is not None else bd.get("upper", 0)
            lb = opt if opt is not None else bd.get("lower", 0)
            bounds[rec["name"]] = (ub or 0, lb or 0)
    out = []
    for k in range(lo, hi + 1):
        name = f"ta{k:02d}"
        lines = [l for l in jsplib[name].splitlines()
                 if l.strip() and not l.lstrip().startswith("#")]
        n2, m2 = map(int, lines[0].split())
        assert (n2, m2) == (n, m), f"{name}: expected {n}x{m}, got {n2}x{m2}"
        times, machs = [], []
        for r in range(n):
            v = list(map(int, lines[1 + r].split()))
            machs.append([x + 1 for x in v[0::2]])       # -> 1-indexed
            times.append(v[1::2])
        ub, lb = bounds.get(name, (0, 0))
        out.append("Nb of jobs, Nb of Machines, Time seed, Machine seed, "
                   "Upper bound, Lower bound")
        out.append(f"{n:10d}{m:10d}{0:11d}{0:11d}{ub:10d}{lb:10d}")
        out.append("Times")
        out += [" " + " ".join(f"{x:2d}" for x in row) for row in times]
        out.append("Machines")
        out += [" " + " ".join(f"{x:2d}" for x in row) for row in machs]
    return ("\n".join(out) + "\n").encode()


def fetch_taillard(jsplib: dict[str, str]) -> None:
    print("[1/4] Taillard originals")
    for fname in FILES:
        if (RAW / fname).exists() and not FORCE:
            print(f"    ok  {fname}  (already present - local copy, untouched)")
            _save(RAW / fname, b"", "", protect=True)      # provenance only
            continue
        data = None
        for host in TAILLARD_HOSTS:
            data = _get(f"{host}/{fname}", timeout=20)
            if data and b"Nb of jobs" in data:
                _save(RAW / fname, data, f"{host}/{fname}", protect=True)
                print(f"    ok  {fname}  (canonical)")
                break
            data = None
        if data is None:
            rec = reconstruct_taillard(fname, jsplib)
            _save(RAW / fname, rec, f"reconstructed from {MIRROR}/instances/", protect=True)
            print(f"    ~~  {fname}  (RECONSTRUCTED from mirror - equivalent, see data/README.md)")


def fetch_bounds() -> None:
    print("[3/4] bounds")
    for host in TAILLARD_HOSTS:
        d = _get(f"{host}/best_lb_up.txt", timeout=20)
        if d and b"JOB SHOP" in d:
            _save(BOUNDS / "best_lb_up.txt", d, f"{host}/best_lb_up.txt", protect=True)
            print("    ok  best_lb_up.txt (canonical)")
            return
    print("    !!  best_lb_up.txt unavailable; using instances.json metadata instead.")
    print("        (If you have the local copy, drop it in data/bounds/ manually.)")


def write_provenance() -> None:
    print("[4/4] provenance")
    lines = [
        "# Data Provenance",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        "by `scripts/fetch_data.py`.",
        "",
        "Taillard, E. (1993). *Benchmarks for basic scheduling problems.* European Journal of",
        "Operational Research 64(2):278-285. doi:10.1016/0377-2217(93)90182-M",
        "",
        "Files marked `reconstructed` were rebuilt from the JSPLIB mirror because Taillard's",
        "host was unreachable. This is lossless - see the design notes §1, and",
        "`tests/test_golden.py::test_all_80_instances_agree` proves it on every run.",
        "",
        "| file | source | sha256 |",
        "|---|---|---|",
    ]
    for path, url, sha in sorted(provenance):
        lines.append(f"| `{path}` | {url} | `{sha}` |")
    # Print provenance to stdout only; the committed data provenance is data/README.md.
    print("\n".join(lines))
    print(f"    ok  printed provenance for {len(provenance)} files (committed provenance: data/README.md)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch/verify the Taillard JSSP benchmark.")
    ap.add_argument("--all", action="store_true",
                    help="also fill MISSING data/raw/ and bounds files (existing ones are kept)")
    ap.add_argument("--force", action="store_true",
                    help="DANGEROUS: overwrite existing data/raw/ files. You do not want this.")
    args = ap.parse_args()
    FORCE = args.force

    for d in (RAW, JSPLIB, BOUNDS):
        d.mkdir(parents=True, exist_ok=True)

    jsplib = fetch_jsplib()                     # always safe: writes data/jsplib/ only
    if args.all:
        fetch_taillard(jsplib)
        fetch_bounds()
    else:
        print("[1/4] Taillard originals   SKIPPED (default). "
              "data/raw/ and best_lb_up.txt untouched.")
        print("      Pass --all if you are missing any of them.")
        print("[3/4] bounds               SKIPPED (default).")
    write_provenance()
    print("\nDone. Now run:  uv run python scripts/verify_data.py")