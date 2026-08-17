# JSSP Tree-Seed Benchmark

Independent, reproducible implementations of two tree-seed metaheuristics on the **Taillard
job-shop scheduling benchmark (JSSP)**, with every published number re-derived from committed data
by a script in this repository.

- **ATSA** — the Advanced Tree-Seed Algorithm of Şahman (2022), reproduced on the paper's full
  40-instance set.
- **DTSA** — the Discrete Tree-Seed Algorithm of Cinar, Korkmaz & Kiran (2020), first validated on
  the symmetric TSP (the problem that paper solves), then ported to the job shop and compared
  against ATSA under an identical evaluation budget.
- **TSA** — the basic Tree-Seed Algorithm, as a shared-core baseline.

This repository is a clean release for engineers who are not Python specialists. If you can install
one tool (`uv`) and run three commands, you can reproduce the headline results.

---

## 1. What this repository is

| | |
|---|---|
| **Algorithms** | ATSA (Şahman 2022, Algorithm 2), DTSA (Cinar 2020), basic TSA baseline |
| **Benchmark** | Taillard job-shop instances ta01–ta80; the paper's comparison set is 40 of them |
| **Reproduced** | ATSA matches the paper's Table 5 on **40/40** instances within **±3.1%** |
| **New** | A like-for-like ATSA-vs-DTSA comparison under equal `MaxFEs`; several documented defects in the source papers; a Numba-optional pure-Python fallback |
| **Not learned** | These are search metaheuristics. All timing is **solve time**, never training time. |

## 2. Headline results

**ATSA reproduction — 40 paper instances** (computed by `scripts/compute_headline.py` from
`results/atsa/atsa_ta01.csv` + `results/atsa/atsa_ta02_ta80.csv` vs `src/atsa_jssp/paper_table5.py`):

| metric (over the 40-instance paper scope) | value |
|---|---|
| instances within ±3.1% of the paper | **40 / 40** |
| max absolute difference | **3.04 %** |
| mean signed difference | **+0.19 %** |
| mean absolute difference | **0.76 %** |
| correlation(D, difference %) | **−0.02** (no size dependence) |

**DTSA-core vs our ATSA — 40 paper instances, sampler C3** (computed by `dtsa/metrics.py` from
`results/dtsa/dtsa_jssp_c3.csv` + `results/atsa/`):

| metric (40 instances, N=40, sampler C3) | value |
|---|---|
| DTSA-core mean advantage vs our ATSA | **+4.06 %** |
| DTSA-core mean advantage vs *published* ATSA | **+3.88 %** |
| instances where DTSA-core wins | **40 / 40** |

> All aggregates above are over the **40-instance** paper scope in `data/paper_scope_40.txt`.
> Instance counts are stated next to every number on purpose — see `results/atsa/README.md`.

## 3. Repository map

```
src/atsa_jssp/     ATSA / TSA algorithm, decoder, operators, CLI  (installable package)
reference/         pure-Python "oracle" implementations the fast code is tested against
dtsa/              DTSA job-shop port, ablation drivers, metrics.py (the number source)
  validation_tsp/  the DTSA TSP validation gate (tsp, two-opt, gate 1)
  tests/           DTSA test suite
data/              Taillard instances (raw/ + jsplib/ cross-check), bounds, tsplib/, paper scope
results/           committed result CSVs: atsa/ dtsa/ ablations/ timing/ gate1/ sweep/
reports/           the ATSA reproduction report and the DTSA report
scripts/           data verification, headline-number computation, optional fetchers
tests/             ATSA golden tests + the paper Table-5 checksum + the scope test
run_atsa_campaign.py   run ATSA across ta02..ta80
```

## 4. Prerequisites — from a machine with nothing installed

This project targets **Python 3.12** (see `.python-version`) and uses **`uv`** to install a pinned,
reproducible environment. `uv` creates the virtual environment and fetches Python for you.

### Windows (PowerShell)
```powershell
# 1. install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# 2. clone
git clone https://github.com/soumysuwas/jssp-tree-seed-benchmark.git
cd jssp-tree-seed-benchmark
# 3. install everything (Python 3.12 + all dependencies), then verify
uv sync
uv run python scripts/verify_data.py     # expect: 80/80 instances agree across formats
```

### macOS (Apple Silicon) / Linux (x86-64 or aarch64)
```bash
# 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. clone
git clone https://github.com/soumysuwas/jssp-tree-seed-benchmark.git
cd jssp-tree-seed-benchmark
# 3. install everything, then verify
uv sync
uv run python scripts/verify_data.py     # expect: 80/80 instances agree across formats
```

> **Intel Macs (x86-64):** the pinned numba 0.67.0 / llvmlite 0.49.0 ship cp312 wheels for
> `win_amd64`, `macosx_12_0_arm64`, and Linux `x86-64`/`aarch64` — but **not** for macOS x86-64.
> So `uv sync` (and `pip install -r requirements.txt`) will try to build numba from source and will
> most likely fail. Numba is optional at runtime, so install the pure-Python deps *without* it and
> run with the fallback:
> ```bash
> python3.12 -m venv .venv && source .venv/bin/activate
> pip install "numpy>=1.26,<2.2" "joblib>=1.4" "pandas>=2.2" "typer>=0.12" "rich>=13.7"
> pip install -e . --no-deps          # install the atsa_jssp package only
> export ATSA_NO_NUMBA=1               # use the pure-Python path
> python scripts/verify_data.py
> ```
> The pure-Python path is slower but produces **identical makespans** (measured — see §9). Apple
> Silicon Macs use the arm64 wheel and need none of this.

Not using `uv`? Install Python 3.12 yourself and `pip install -r requirements.txt`.

**If Numba fails to install** (rare; it is a compiled package): the code still runs. It falls back
to pure Python automatically and prints a warning. See §10.

## 5. Quickstart — one instance, a few minutes

```bash
uv run python -m atsa_jssp.cli run --instance ta01 --runs 2 --jobs 2 --out results/quickstart_ta01.csv
```

Real output (the first run includes Numba compile time):

```
ATSA  1 instance(s) x 2 runs  jobs=2
  rand_lt_st / seed / continuous / strict_fe_cap=False
instance  runs   mean    med  min  max   std  wall_s_mean  fes_min  fes_max
    ta01     2 1465.5 1465.5 1457 1474 12.02         1.41   225257   225304

wrote results\quickstart_ta01.csv  (2 rows)
```

The package also installs a console script, so `uv run atsa run --instance ta01 --runs 2` is
equivalent to `uv run python -m atsa_jssp.cli run --instance ta01 --runs 2`. Both forms are used
interchangeably below.

`min = 1457` is the best makespan found; the paper's best for ta01 is 1347. Two seeds is just a
smoke test — the paper uses 20.

## 6. Full reproduction

Runtimes below are wall-clock on an Intel i9 (24 logical cores), Windows 11, Python 3.12 + Numba.
Times scale with instance size; ta71–ta80 (D=2000) dominate.

| Step | Command | What it does | Writes |
|---|---|---|---|
| Verify data | `uv run python scripts/verify_data.py` | 80/80 cross-format check (seconds) | — |
| Tests | `uv run pytest -q` | golden tests, Table-5 checksum, scope test (~1–2 min) | — |
| ATSA, one instance | `uv run python -m atsa_jssp.cli run --instance ta01 --runs 20` | 20 seeded runs | `results/atsa_ta01.csv` (scratch, top-level) |
| ATSA, 40 paper instances | `uv run python -m atsa_jssp.cli run --all --runs 20 --jobs 8` | the paper set (ta71–75 last) | `results/atsa_full.csv` (scratch, top-level) |
| ATSA, ta02–ta80 campaign | `uv run python run_atsa_campaign.py --jobs 8` | broad coverage, 79 instances (tens of minutes) | `results/atsa/atsa_ta*.csv`, `results/atsa/atsa_ta02_ta80.csv` |
| DTSA job-shop (C1 / C3) | `uv run python -u dtsa/run_jssp.py --jobs 8` / `run_jssp_c3.py` | DTSA-core runs | `results/dtsa/dtsa_jssp*.csv` |
| DTSA summary table | `uv run python dtsa/make_dtsa_table5.py` | per-instance summary | `results/dtsa/dtsa_table5.csv` |
| TSP validation gate | `uv run python dtsa/validation_tsp/run_gate1.py --jobs 8` | the gate that did **not** pass (see `results/gate1/README.md`) | `results/gate1/gate1.csv` |
| Headline numbers | `uv run python scripts/compute_headline.py` | prints the §2 ATSA numbers | — |
| DTSA numbers | `uv run python dtsa/metrics.py` | prints the §2 DTSA numbers | — |

> **Where files land.** The `cli run` command writes to the top level of `results/` by default
> (e.g. `results/atsa_ta01.csv`). That location is **scratch** — it is gitignored (`/results/*.csv`)
> so your runs never overwrite the committed data, which lives one level down in `results/atsa/`.
> To write a file directly comparable to a committed one, pass `--out`, e.g.
> `--out results/atsa/atsa_ta01.csv`. Only `run_atsa_campaign.py` writes into `results/atsa/` itself.

Ablation and timing drivers are listed in `reports/DTSA_report.md` §7 and in the per-folder READMEs
under `results/`.

## 7. How to read the results

Each row of an ATSA/DTSA CSV is **one run** of one instance with one seed.

| Column | Meaning |
|---|---|
| `instance` | e.g. `ta01` |
| `n`, `m`, `D` | jobs, machines, dimension `D = n × m` |
| `seed` | RNG seed (0..runs−1); the run is fully determined by (instance, seed, config) |
| `cmax` | **makespan** — the finish time of the last operation; **lower is better** |
| `fes_used` / `max_fes` | function evaluations used / the budget `D × 1000` |
| `wall_s` | **solve time** in seconds for that run (not training time) |
| `git_sha` | the internal development revision that produced the row (provenance only; these hashes are from the private development repo and are **not** resolvable in this repository). The committed ATSA results span two such revisions (`53421d8`, `be99d56`); the run configuration — `algorithm, st_sense, operator_space, branch_granularity, N, ST, L, U, max_fes/D` — is verified identical across both. |

Definitions:
- **Makespan (Cmax):** the total time to complete all jobs on the schedule the algorithm returns.
- **Gap to best-known:** `(cmax − best_known) / best_known`, using Taillard's bounds in
  `data/bounds/best_lb_up.txt`. The paper reports makespans, not gaps; we follow the paper.
- **Elapsed / solve time:** wall-clock to run the search under a fixed evaluation budget. There is
  **no training phase** — nothing is learned.

Comparing against the paper: the paper's Table 5 is committed verbatim in
`src/atsa_jssp/paper_table5.py`. `scripts/compute_headline.py` reads it and our CSVs and prints the
per-instance and aggregate differences, over the 40-instance scope only.

`results/dtsa/dtsa_table5.csv` is a **per-instance descriptive summary** (runs/mean/median/min/max/std)
prepared as input for any separate statistical analysis. It is **not** itself a statistical test —
see §11.

## 8. Configuration

| Setting | Default | Where |
|---|---|---|
| RNG seeds | 0 .. runs−1 | `--runs` on the CLI; recorded on every CSV row |
| Function-evaluation budget | `MaxFEs = D × 1000` | `--fe_multiplier` (CLI), `Config` in `src/atsa_jssp/atsa.py` |
| Population size `N` | 40 | `Config.N` |
| Search tendency `ST` | 0.2 | `Config.ST` |
| Worker count | logical cores | `--jobs` (CLI / campaign / DTSA drivers) |

The default worker count is `os.cpu_count()`. Any value is memory-clamped by
`experiment.py::safe_n_jobs()` so a large instance cannot exhaust RAM.

## 9. Hardware and runtime expectations

- **One seed of every one of the 40 instances, serial (`--jobs 1`):** ≈ 29 minutes (DTSA-core
  timing table, `results/timing/`). ATSA is faster.
- **Decoder throughput:** the Numba decoder runs the ATSA hot loop ≈ **69×** faster than the pure
  Python oracle (compile time excluded).
- **ATSA vs DTSA per run (equal `MaxFEs`):** DTSA-core is on average **~2.1× slower per run** than
  ATSA (**1.12×** at the largest instance ta71), and ATSA is faster on **40/40** instances. This is
  an **implementation** difference (ATSA's loop is Numba-compiled; DTSA-core's is plain NumPy for
  readability), **not** an algorithmic one — the fair basis is the identical evaluation budget.
  Source: `dtsa/metrics.py` over `results/timing/`.
- **Running many jobs at once (contention):** measured ATSA per-run slowdown is **1.08×** at 8
  concurrent jobs and **1.60×** at 24 (source: `results/timing/atsa_timing_contention.csv`).
  Throughput still rises with more jobs; individual runs just get slower.

**Determinism (measured, not assumed).** Makespans are fully determined by (instance, seed, config):
- worker count does not change them — ta01 seeds 0–3 give `[1457, 1474, 1643, 1415]` at both
  `--jobs 1` and `--jobs 4` (identical);
- Numba on vs off does not change them — the same seeds give the same makespans in both modes (§10).

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| **Numba fails to install or import** | The code auto-falls back to pure Python and warns. To force it: set `ATSA_NO_NUMBA=1`. Results are identical (verified); runs are much slower. |
| **`uv sync` fails building numba on an Intel Mac (x86-64)** | Expected: the pinned numba/llvmlite have no macOS x86-64 wheel (only arm64). Install the runtime deps without numba and use the fallback — see the Intel-Macs note in §4. Apple Silicon is unaffected. |
| **Out of memory on ta71–ta80** | Each worker is a full process (~0.9 GB with Numba). Lower `--jobs`. `safe_n_jobs()` already clamps to available memory. |
| **Windows/macOS: process spawn errors or re-running forever** | Always launch via the provided scripts/CLI; every entry point is guarded by `if __name__ == "__main__":`, which the spawn start method requires. |
| **`80/80` check fails / missing data** | Ensure `data/raw/` and `data/jsplib/` are present (they ship with the repo). Re-run `scripts/verify_data.py`. |
| **A run takes much longer than expected** | The first run pays Numba's one-time compile cost. Large D and high `--jobs` contention both slow individual runs; see §9. |

## 11. Known limitations, and defects we found in the source papers

Stated as our findings, with what we did instead:

- **The paper's improvement percentages use a non-standard denominator** (dividing by the improved
  value, not the baseline), which inflates them. Its own PSO figure proves it. We report differences
  the conventional way (ATSA report §6.1).
- **A third of ATSA's evaluation budget does nothing.** Branches C and D consume ~35.7% of the
  budget and produced zero best-solution updates at either D=225 or D=2000 (ATSA report §6.2).
- **The DTSA TSP validation gate did not pass** its pre-registered tolerance. We document it in
  `results/gate1/` rather than dropping it, and carry the caveat through the DTSA report.
- **We caught a transcription error in our own reference column** that had produced a false
  "size-dependent drift". The paper's own AVG row is a checksum; it is now a permanent test
  (`tests/test_paper_table5.py`).
- **No inferential statistics are claimed as this repository's output** — see below.

### What this repository does **not** contain
- **No source-paper PDFs.** They are copyrighted; cite them from the DOIs in §12. The one place the
  papers' numbers appear is `src/atsa_jssp/paper_table5.py` / `dtsa/dtsa_tables.py`, transcribed
  offline and checksum-tested.
- **No inferential statistical tests.** This repository does checksum/gate validation and descriptive
  statistics (means, differences, reproducibility). It does **not** compute or claim Wilcoxon,
  Friedman, p-values, or confidence intervals. `dtsa_table5.csv` is descriptive input for a separate
  analysis, not a test result.
- **No MATLAB implementation.** Everything here is Python.
- **No proprietary, plant, or production data.** Only the public Taillard and TSPLIB benchmarks.

Every result CSV in `results/atsa/`, `results/dtsa/`, `results/ablations/`, `results/timing/`, and
`results/gate1/` is regenerated by a script in this repository. The only exception is the eight
archived operator-sweep CSVs in `results/sweep/` — see `results/sweep/README.md`.

## 12. References

- M. A. Şahman (2022). *Advanced Tree-Seed Algorithm for Large Sized Job Shop Scheduling Problems.*
  Gazi Journal of Engineering Sciences 8(2):201–214. doi:10.30855/gmbd.0705004. (Open access, CC-BY.)
- A. C. Cinar, S. Korkmaz, M. S. Kiran (2020). *A discrete tree-seed algorithm for solving symmetric
  traveling salesman problem.* Engineering Science and Technology, an International Journal
  23(4):879–890. doi:10.1016/j.jestch.2019.11.005.
- M. S. Kiran (2015). *TSA: Tree-Seed Algorithm for continuous optimization.* Expert Systems with
  Applications 42(19):6686–6698.
- E. Taillard (1993). *Benchmarks for basic scheduling problems.* European Journal of Operational
  Research 64(2):278–285. doi:10.1016/0377-2217(93)90182-M.

## 13. Authorship and contact

**Soumy Suwas** — M.Tech (AI), Indian Institute of Technology Hyderabad.
Issues and questions: please use the GitHub repository's issue tracker.
Licensed under the MIT License (see `LICENSE`).
