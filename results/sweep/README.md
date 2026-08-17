# results/sweep/

Two kinds of archived diagnostic runs from the ATSA study.

## `tsa_bisect.csv` — regenerable

A TSA bisection over the instance-size range (180 runs). Regenerate:

```bash
uv run python run_tsa_bisect.py
```

## `rand_lt_st_*.csv` / `st_lt_rand_*.csv` (8 files) — **archived, generator not included**

These are the output of an ad-hoc parameter sweep over ATSA's configuration flags
(`st_sense` × `branch_granularity` × `operator_space`). The exact driver script that produced these
eight files was a one-off and is **not part of this repository**, so there is no committed script
that reproduces them byte-for-byte. They are kept as an **archived artifact**.

You can run equivalent sweeps yourself with the CLI, e.g.:

```bash
uv run python -m atsa_jssp.cli run --all --st_sense st_lt_rand --branch_granularity dimension \
    --operator_space continuous --out results/sweep/st_lt_rand_dimension_continuous.csv
```

but freshly generated files are not guaranteed identical to the archived ones, because the original
run configuration (seed set, run count) was not recorded alongside them. Treat the archived CSVs as
historical evidence, not as a regeneration target.
