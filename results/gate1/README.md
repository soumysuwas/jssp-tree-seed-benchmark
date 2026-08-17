# results/gate1/ — TSP validation gate (did not pass)

Before porting DTSA to the job shop, the DTSA implementation was validated on the **symmetric TSP**,
which is the problem the source paper (Cinar et al. 2020) actually solves. "Gate 1" was a
**pre-registered** check: reproduce the paper's published TSA/DTSA tour lengths on the standard
TSPLIB instances within a stated tolerance.

**Gate 1 did not pass its pre-registered criterion** — the anchor row deviated by more than the
tolerance we had fixed in advance. That result is *why this folder exists*: it is documented rather
than quietly dropped, because a validation gate that fails is evidence about the implementation, not
something to hide. The subsequent job-shop work is reported with that caveat throughout the DTSA
report.

| File | Contents |
|---|---|
| `gate1.csv` | Per-configuration TSP runs (operator × source tree × sampler × seeds). |
| `gate1_summary.csv` | Aggregated per-configuration means vs the paper's Table 1. |

Regenerate:

```bash
uv run python dtsa/validation_tsp/run_gate1.py --jobs 8      # writes gate1.csv
uv run python dtsa/validation_tsp/analyse_gate1.py           # writes gate1_summary.csv
```
