# results/atsa/

ATSA runs on the Taillard job-shop instances, 20 independent seeds (0–19) per instance.

| File | Contents |
|---|---|
| `atsa_ta01.csv` | ta01 alone, 20 seeds. Reproduce with `python -m atsa_jssp.cli run --instance ta01 --runs 20 --out results/atsa/atsa_ta01.csv` (the 2-run quickstart in the README does **not** reproduce this file). |
| `atsa_ta02_ta80.csv` | The combined campaign output: **79 instances, ta02–ta80**. |
| `atsa_ta<NN>.csv` | Per-instance files, ta02–ta80. |

`atsa_ta01.csv` + `atsa_ta02_ta80.csv` together cover ta01–ta80.

## Scope — read this before quoting any aggregate

The **paper comparison scope is 40 instances**: ta01–05, ta11–15, ta21–25, ta31–35, ta41–45,
ta51–55, ta61–65, ta71–75 (listed in `../../data/paper_scope_40.txt`). Every ATSA-vs-paper number
in the README and the reports is computed over exactly those 40 instances by
`../../scripts/compute_headline.py`.

The other 40 instances (ta06–10, ta16–20, …, ta76–80) were run for **additional coverage only**.
They are **not part of any paper comparison** — the paper never reports them.

`atsa_ta02_ta80.csv` holds 79 instances. No aggregate may be computed over it alone without
stating the instance count it covers; the paper-scope aggregates always filter to the 40 above.

**Columns:** `instance, n, m, D, algorithm, seed, cmax, fes_used, max_fes, iters, wall_s, …` plus
per-branch FE counts. `cmax` is the makespan; one row per (instance, seed).

**On `git_sha`.** This column records the internal development revision that produced each row. The
hashes are from the private development repo and are **not** resolvable here; they are retained for
provenance only. These committed ATSA results span two revisions (`53421d8`, `be99d56`); the run
configuration (`algorithm, st_sense, operator_space, branch_granularity, N, ST, L, U, max_fes/D`) is
verified identical across both.
