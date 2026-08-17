# Benchmark data — provenance and checksums

## What is here

| Folder | Contents |
|---|---|
| `raw/` | The **Taillard job-shop instances** in Taillard's own format: 8 files, 10 instances each = ta01–ta80. Machines are 1-indexed; the Times block and the Machines block are separate. |
| `bounds/` | `best_lb_up.txt` — Taillard's best-known lower/upper bounds (updated edition). Bounds come from here, never from the stale `Upper bound` field inside the instance headers. |
| `jsplib/` | A cross-check mirror of the same 80 instances from JSPLIB (`tamy0612/JSPLIB`). Used only to verify `raw/`; see below. |
| `tsplib/` | A few TSPLIB instances (berlin52, kroA–E100) used by the DTSA TSP validation gate — see `../dtsa/validation_tsp/` and `../results/gate1/`. Not part of the job-shop benchmark. |
| `paper_scope_40.txt` | The 40-instance comparison scope of Şahman (2022), derived programmatically from `../src/atsa_jssp/paper_table5.py`. |

## Source

The Taillard job-shop benchmark is a standard, public benchmark:

> Taillard, E. (1993). *Benchmarks for basic scheduling problems.* European Journal of
> Operational Research 64(2):278–285. doi:10.1016/0377-2217(93)90182-M

`raw/` and `bounds/` hold the complete Taillard set (8 files × 10 instances = ta01–ta80).
`jsplib/` was fetched once from `https://raw.githubusercontent.com/tamy0612/JSPLIB/master/instances/<name>`
purely to cross-check `raw/`. `tests/test_golden.py` and `scripts/verify_data.py` prove the two
sources describe the same 80 instances operation-by-operation (80/80, modulo 0- vs 1-indexed
machines).

Verify locally:

```bash
uv run python scripts/verify_data.py       # -> 80/80 instances agree across formats
```

## SHA256 — Taillard originals (`raw/` + `bounds/`)

| file | sha256 |
|---|---|
| `raw/tai15_15.txt`  | `37d57f0cc7e7837c1a11cdf49e40aef2f1f5551ef737daee331f7df742876d2e` |
| `raw/tai20_15.txt`  | `b7ae82a2de623754dab9769b5d90ab669ad0f63361c2603077a157783d3cca29` |
| `raw/tai20_20.txt`  | `44008d341c28c16b812e85b8ae1d9a6ecd48ac637a13058b70a66653ecac4471` |
| `raw/tai30_15.txt`  | `b6ad3a3537bcacc73ea9d360b0d501a4d9d4911f17290a6413fec49f75b402c9` |
| `raw/tai30_20.txt`  | `1ac92c6c42725173c0fdbe2fcd735f60c01626385c118abe797bf12a3d246432` |
| `raw/tai50_15.txt`  | `f7d31bc9f4451c555438349e88c46a5e90a777b3347d382693a52399f7727fe1` |
| `raw/tai50_20.txt`  | `3cb6f5ec7e528c433243ac37a7452a79808a73f78bc45bc23f855dd3792d911f` |
| `raw/tai100_20.txt` | `da1fc2d76c86105a65ac658d5331b513c9c13b8462e634793a8ad58859b2a58d` |
| `bounds/best_lb_up.txt` | `cbf725645da2e5886cbefebe790ea1f531f1fce1fba1ba803ab648a62a2bf464` |

## SHA256 — TSPLIB instances (`tsplib/`, for the DTSA validation gate)

| file | sha256 |
|---|---|
| `tsplib/berlin52.tsp`     | `8496a5838133e3c3eb25a9fa8893b1ac302593fc8dc325350643a0cabd8abadc` |
| `tsplib/berlin52.opt.tour`| `8f3915f397dcdbf9f07bd570fc41faa7e9b3d27391537f048164b2f937d1ccc9` |
| `tsplib/kroA100.tsp`      | `e103100c1cf31dfc06be95a9d04011b5a8753bb65a3339594ca34404e574bdf5` |
| `tsplib/kroA100.opt.tour` | `65499d540c5aa7e90750065a5c6ff82f9774f284facf9d29af0146bce4332dd1` |
| `tsplib/kroB100.tsp`      | `283d8c912e3334deea76cc9fe95e915d09111979e7753d0affaf14d9aa21cdbe` |
| `tsplib/kroC100.tsp`      | `aef750b8051e011df3b0fae21f4d71273cf129bcf7f2a9b697c2406bcaec68d2` |
| `tsplib/kroD100.tsp`      | `e2297d9c40abc8341792e7b785e7a9c04ed21dfda0c667d9388888b4ba8cb52d` |
| `tsplib/kroE100.tsp`      | `de5a20663b3903407150d1798c9be517d2df2030d7a910ddc9fc3c024a5c3bb4` |

The 80 `jsplib/` cross-check files are verified against `raw/` by the tests above rather than by a
checksum table here. To re-fetch them, see `scripts/fetch_data.py` (optional; the data is already
committed).
