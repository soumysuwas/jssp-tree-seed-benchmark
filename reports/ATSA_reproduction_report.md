# Reproducing ATSA on the full Taillard JSSP benchmark

**An independent reproduction of Şahman (2022), all 40 paper instances, 20 seeded runs each.**

Soumy Suwas — M.Tech (AI), Indian Institute of Technology Hyderabad

> M. A. Şahman (2022), "Advanced Tree-Seed Algorithm for Large Sized Job Shop Scheduling
> Problems", *Gazi Journal of Engineering Sciences* 8(2):201–214, doi:10.30855/gmbd.0705004.
> Open access, CC-BY.

Every number attributed to the paper below comes from `src/atsa_jssp/paper_table5.py`, a
committed transcription of Table 5, and is checked against the paper's own AVG row on every
test run — see §9, which explains why that matters more than it sounds. The paper's PDF is
not redistributed here; cite it from the DOI above.

---

## Headline

1. **ATSA reproduces, uniformly across the whole size range.** 40/40 instances within **±3.1%** of
   Table 5; mean difference **+0.19%**; **no size dependence** (corr(D, diff%) = **−0.02**). §5.
2. **The paper's headline improvement figures use a non-standard denominator that inflates all of
   them.** It divides by the *improved* value, not the baseline. Its own PSO figure proves it to
   the digit. Conventionally, 30.33% → **23.25%** and 12.04% → **10.75%**. §6.1.
3. **A third of ATSA's function-evaluation budget does nothing.** Branches C and D (Eq.3 and
   Eq.4) consume **35.7%** of every budget, accept **~0.08%** of seeds, and produced **zero**
   best-solution updates at either D=225 or D=2000. §6.2.
4. **Basic TSA also reproduces** across the size range, confirming the shared core. §8.
5. **We caught a transcription error in our own reference column that had produced a
   statistically overwhelming — and entirely false — "size-dependent drift" finding.** The
   paper's own AVG row is a checksum that would have caught it immediately. It is now a
   permanent test. §9.

---

## 1. Scope — what "the benchmark" actually is

The paper's Table 3 defines its test set as **40 instances**: the first five of each of eight size
groups, ta01–05, ta11–15, ta21–25, ta31–35, ta41–45, ta51–55, ta61–65, ta71–75. The largest is
**ta75**. Dimension `D = n × m` runs from 225 to 2000.

The Taillard distribution contains the complete set, **ta01–ta80**, because
that is how Taillard distributes it (8 files × 10 instances). **The other 40 instances are not in
the paper** and no comparison against it is possible for them. We ran them anyway as reference
data — see §12.

| Group | n × m | D | Instances in file | Paper uses |
|---|---|---|---|---|
| tai15_15 | 15×15 | 225 | ta01–ta10 | ta01–05 |
| tai20_15 | 20×15 | 300 | ta11–ta20 | ta11–15 |
| tai20_20 | 20×20 | 400 | ta21–ta30 | ta21–25 |
| tai30_15 | 30×15 | 450 | ta31–ta40 | ta31–35 |
| tai30_20 | 30×20 | 600 | ta41–ta50 | ta41–45 |
| tai50_15 | 50×15 | 750 | ta51–ta60 | ta51–55 |
| tai50_20 | 50×20 | 1000 | ta61–ta70 | ta61–65 |
| tai100_20 | 100×20 | 2000 | ta71–ta80 | ta71–75 |

## 2. Provenance — the data was already complete

**The Taillard distribution already contains all 8 files plus `best_lb_up.txt`.** Nothing needed
to be sourced from GitHub. This corrects an earlier assumption that some files were missing, where the
group believed only part of the set had been shared.

Verification, not assumption:

- All 80 instances parsed from the Taillard-original format and cross-checked
  **operation-by-operation** against the JSPLIB mirror (`tamy0612/JSPLIB`): **80/80 identical**.
- SHA256 of all 89 data files recorded in `data/README.md`.
- Reproduce with `uv run python scripts/verify_data.py` → `80/80 instances agree across formats`.

> The `Upper bound` column *inside* the instance files is stale (1993–97) and disagrees with
> current optima for ta03/ta04/ta05. Bounds come from `best_lb_up.txt` (Taillard's own, updated
> 2013), never from the instance headers.

## 3. Setup — exactly what was run

| | |
|---|---|
| Algorithm | ATSA (Şahman 2022, Algorithm 2 / Figure 6), literal transcription |
| Population `N` | 40 |
| Search tendency `ST` | 0.2 |
| Seeds per tree `NS` | random in [L,U] = [4,10] (L=0.1N, U=0.25N) |
| Search range | [-5, 5] |
| Budget | **MaxFEs = D × 1000**, counted in function evaluations, not iterations |
| Runs per instance | **20 independent** |
| **RNG seeds** | **0–19, fixed and recorded on every CSV row** |
| Encoding | Bean random-key → Eq.2 `S = (τ mod n) + 1` → semi-active decoder |
| Local search | **None.** No 2-opt, no left-shifting — the paper has none |

Hardware / software: Intel i9 (24 logical cores), 32 GB, Windows 11, Python 3.12 + Numba,
process-parallel over runs (`joblib`, loky), `NUMBA_NUM_THREADS=1` per worker.

Decoder throughput: **276,564 FE/s** on ta01, single core (3.62 µs/FE) — **69× the NumPy oracle**
(249.4 µs/FE), compile time excluded.

Cost: **1,600 runs, 28.5 core-hours of CPU, 87 minutes of wall clock** (measured from the first
and last run timestamps in the CSV). The TSA bisection (§8) adds 180 runs.

Every run records its seed, final FE count, full config and the git SHA that produced it.
**Re-running the CSV reproduces it bit-for-bit** — verified by re-running Gate 4 end-to-end.

> **Practical note for anyone re-running this.** `--jobs -1` is *not* a safe default here. Each
> worker is a full Python process carrying its own Numba/LLVM runtime (~0.9 GB resident — the
> algorithm's own arrays are ~0.6 MB and irrelevant). This machine has 31.8 GB RAM but a
> **33.8 GB commit limit** (≈ no pagefile headroom), so 24 workers on D=2000 exceed it and
> Windows kills workers or destabilises. `experiment.py::safe_n_jobs()` now clamps the worker
> count to what memory can commit (24 → 13 here).

## 4. Verification — six gates with known answers

Each has an answer that does not depend on our implementation.

| # | Gate | Result |
|---|---|---|
| 1 | The paper's own 2×3 worked example (Table 1/2, Figure 1) | ranks `[2,5,6,3,1,4]`, Eq.2 `[1,2,1,2,2,1]`, Gantt exact, **Cmax = 22** ✓ |
| 2 | Operators reproduce Figures 3, 4, 5 | swap → `1 2 6 4 5 3 7`, symmetry → `1 6 5 4 3 2 7`, shift → `1 3 4 5 2 6 7` ✓ |
| 3 | Numba twin == NumPy oracle | agree on **30,000** random inputs, 3 instances ✓ |
| 4 | FE accounting | measured **E[FE/seed] = 1.398–1.402** vs **1.4** predicted analytically ✓ |
| 5 | Data integrity | **80/80** instances identical across two independent formats ✓ |
| 6 | **Table 5 transcription** | **all 20 columns match the paper's own AVG row** (`tests/test_paper_table5.py`) ✓ |

Gate 4 in detail — the branch structure was counted, not assumed:

| Branch | Predicted | Measured | FEs each |
|---|---|---|---|
| A — swap on Best | 10.0% | 9.8% | 1 |
| B — symmetry + shift | 40.0% | 39.2% | **2** |
| C — Eq.3 | 37.5% | 38.0% | 1 |
| D — Eq.4 | 12.5% | 13.0% | 1 |

The identity `A + 2B + C + D == FEs − N` is **asserted on every run**. Final FE counts land in
[225,020, 225,400] against a 225,000 budget — the documented overshoot from checking the cap at
the `while` (Algorithm 2 line 1), never mid-loop.

**Gate 6 is new, and it exists because we failed it.** See §9.

## 5. Results — all 40 paper instances, ours vs Table 5

20 runs each, seeds 0–19. `diff%` is on the mean; positive = our makespan is larger (worse).

**40/40 instances land within ±3.1% of the paper. Mean difference +0.19%. Largest |diff| 3.04%.**

| inst | D | our mean | paper mean | diff% | z | our med | paper med | our min | paper min |
|---|---|---|---|---|---|---|---|---|---|
| ta01 | 225 | 1460.3 | 1444.8 | +1.07 | +1.3 | 1459.5 | 1445.0 | 1382 | 1347 |
| ta02 | 225 | 1440.8 | 1435.0 | +0.40 | +0.4 | 1415.0 | 1421.0 | 1356 | 1374 |
| ta03 | 225 | 1433.1 | 1430.6 | +0.17 | +0.3 | 1443.5 | 1433.5 | 1362 | 1353 |
| ta04 | 225 | 1418.3 | 1438.1 | -1.38 | -1.5 | 1427.5 | 1432.5 | 1309 | 1366 |
| ta05 | 225 | 1442.0 | 1448.5 | -0.45 | -0.6 | 1428.0 | 1448.0 | 1351 | 1344 |
| ta11 | 300 | 1670.9 | 1663.7 | +0.43 | +0.6 | 1676.0 | 1661.5 | 1592 | 1581 |
| ta12 | 300 | 1658.6 | 1685.6 | -1.60 | -2.2 | 1662.0 | 1678.5 | 1573 | 1566 |
| ta13 | 300 | 1651.1 | 1638.8 | +0.75 | +0.8 | 1646.0 | 1647.5 | 1552 | 1517 |
| ta14 | 300 | 1598.6 | 1589.9 | +0.55 | +0.6 | 1602.5 | 1579.5 | 1471 | 1518 |
| ta15 | 300 | 1633.8 | 1685.0 | **-3.04** | -4.0 | 1618.0 | 1676.0 | 1554 | 1607 |
| ta21 | 400 | 2033.1 | 1990.6 | +2.14 | +2.8 | 2032.5 | 1985.5 | 1936 | 1919 |
| ta22 | 400 | 1953.5 | 1944.3 | +0.48 | +0.6 | 1948.5 | 1924.0 | 1849 | 1843 |
| ta23 | 400 | 1925.3 | 1902.1 | +1.22 | +1.4 | 1927.5 | 1906.0 | 1816 | 1791 |
| ta24 | 400 | 1975.5 | 1994.2 | -0.94 | -1.2 | 1974.5 | 2005.5 | 1818 | 1872 |
| ta25 | 400 | 1951.0 | 1934.3 | +0.87 | +1.6 | 1951.5 | 1935.0 | 1851 | 1825 |
| ta31 | 450 | 2147.8 | 2152.9 | -0.24 | -0.3 | 2169.5 | 2151.5 | 2028 | 2050 |
| ta32 | 450 | 2277.2 | 2242.8 | +1.54 | +2.1 | 2291.5 | 2245.5 | 2131 | 2153 |
| ta33 | 450 | 2268.9 | 2250.9 | +0.80 | +1.5 | 2271.5 | 2238.5 | 2170 | 2154 |
| ta34 | 450 | 2254.4 | 2234.5 | +0.89 | +1.2 | 2249.0 | 2219.5 | 2136 | 2171 |
| ta35 | 450 | 2369.4 | 2351.7 | +0.75 | +1.3 | 2369.5 | 2347.5 | 2267 | 2214 |
| ta41 | 600 | 2575.0 | 2571.9 | +0.12 | +0.2 | 2571.5 | 2587.0 | 2467 | 2395 |
| ta42 | 600 | 2540.8 | 2520.4 | +0.81 | +1.2 | 2542.5 | 2504.0 | 2420 | 2361 |
| ta43 | 600 | 2431.4 | 2424.7 | +0.28 | +0.3 | 2400.0 | 2404.5 | 2284 | 2312 |
| ta44 | 600 | 2573.1 | 2544.5 | +1.12 | +1.7 | 2563.0 | 2532.5 | 2439 | 2382 |
| ta45 | 600 | 2498.8 | 2484.9 | +0.56 | +1.2 | 2505.0 | 2482.5 | 2406 | 2366 |
| ta51 | 750 | 3344.4 | 3337.2 | +0.22 | +0.4 | 3326.5 | 3331.0 | 3231 | 3243 |
| ta52 | 750 | 3343.9 | 3371.4 | -0.82 | -1.6 | 3349.0 | 3360.0 | 3185 | 3235 |
| ta53 | 750 | 3095.9 | 3136.2 | -1.28 | -3.4 | 3091.5 | 3122.5 | 3012 | 3023 |
| ta54 | 750 | 3208.7 | 3206.9 | +0.06 | +0.1 | 3214.0 | 3206.0 | 3030 | 3056 |
| ta55 | 750 | 3260.8 | 3261.3 | -0.02 | -0.0 | 3243.5 | 3264.0 | 3156 | 3145 |
| ta61 | 1000 | 3522.5 | 3521.7 | +0.02 | +0.1 | 3536.5 | 3527.0 | 3393 | 3385 |
| ta62 | 1000 | 3581.8 | 3548.0 | +0.95 | +2.3 | 3575.5 | 3563.5 | 3481 | 3365 |
| ta63 | 1000 | 3351.9 | 3322.3 | +0.89 | +1.7 | 3326.5 | 3321.5 | 3264 | 3212 |
| ta64 | 1000 | 3319.3 | 3333.8 | -0.43 | -0.8 | 3324.0 | 3342.5 | 3211 | 3239 |
| ta65 | 1000 | 3380.0 | 3371.9 | +0.24 | +0.4 | 3364.0 | 3367.0 | 3257 | 3277 |
| ta71 | 2000 | 6380.6 | 6441.0 | -0.94 | -3.0 | 6393.0 | 6448.5 | 6235 | 6255 |
| ta72 | 2000 | 5957.4 | 5910.9 | +0.79 | +2.2 | 5972.0 | 5918.0 | 5756 | 5680 |
| ta73 | 2000 | 6549.8 | 6527.0 | +0.35 | +0.9 | 6543.0 | 6521.0 | 6390 | 6324 |
| ta74 | 2000 | 5970.8 | 5987.3 | -0.28 | -1.0 | 5971.5 | 6013.0 | 5831 | 5797 |
| ta75 | 2000 | 6317.4 | 6293.1 | +0.39 | +1.1 | 6310.0 | 6282.0 | 6126 | 6056 |

**By size — there is no size dependence:**

| D | instances | mean z | mean diff% |
|---|---|---|---|
| 225 | 5 | −0.0 | −0.04 |
| 300 | 5 | −0.8 | −0.58 |
| 400 | 5 | +1.0 | +0.75 |
| 450 | 5 | +1.2 | +0.75 |
| 600 | 5 | +0.9 | +0.58 |
| 750 | 5 | −0.9 | −0.37 |
| 1000 | 5 | +0.7 | +0.33 |
| 2000 | 5 | **+0.1** | **+0.06** |

**corr(D, diff%) = −0.02.** 32/40 instances have |z| ≤ 2. The agreement at D=2000 (the paper's
own headline regime — the title is *"for Large Sized Job Shop Scheduling Problems"*) is as good
as at D=225.

**One honest qualification.** 28 of 40 instances are *slightly* worse than the paper (sign test
p = 0.017), so there is a mild systematic tilt. Its magnitude is **+0.19%** — negligible against
the ±5% that different RNG, MATLAB-vs-Python and the documented ambiguities can account for
(`06_VALIDATION.md` §4). We report it rather than round it away, but it does not support any
claim about the algorithm.

**This is a clean, uniform reproduction of Table 5.**

## 6. Findings about the paper

These come from our own instrumentation and from the paper's own arithmetic. Neither depends on
the comparison in §5.

### 6.1 The headline improvement figures use a non-standard denominator

The paper states ATSA improves on PSO by **12.04%** and on TSA by **30.33%** (verbatim, abstract
and §5). Neither follows from its own Table 5 by the conventional definition. Both follow if
improvement is computed as `(baseline − ATSA) / ATSA` — dividing by the **improved** value rather
than the baseline:

| vs | Table 5 AVG | `(b−A)/A` (the paper's) | `(b−A)/b` (conventional) | paper states |
|---|---|---|---|---|
| PSO | 3181.27 | **12.04%** | 10.75% | **12.04%** |
| TSA | 3699.67 | **30.30%** | **23.25%** | **30.33%** |

**The PSO figure identifies the formula beyond doubt: 12.04% reproduces to the digit, while the
conventional denominator gives 10.75%.** The same formula gives TSA 30.30% against the stated
30.33% — the 0.03 pp is rounding in the AVG row itself.

The arithmetic is internally consistent. It is the **denominator that is non-standard, and it is
never stated.** It inflates every headline number in the abstract and conclusion. Conventionally,
the paper's own AVG row gives **23.25%** over TSA and **10.75%** over PSO.

### 6.2 A third of ATSA's budget does nothing

Instrumented runs, ta01 (D=225) vs ta71 (D=2000), one run each:

| Branch | FE share (ta01 → ta71) | acceptance (ta01 → ta71) | became Best (ta01 → ta71) |
|---|---|---|---|
| A — swap on Best | 7.2% → 7.2% | 1.13% → 0.43% | 10 → 41 |
| B — symmetry + shift | 57.1% → 57.1% | 1.53% → 0.71% | 37 → 186 |
| **C — Eq.3** | **26.7% → 26.8%** | **0.08% → 0.01%** | **0 → 0** |
| **D — Eq.4** | **9.0% → 8.9%** | **0.08% → 0.01%** | **0 → 0** |

**Branches C and D — the classic TSA equations Eq.3 and Eq.4 — consume 35.7% of every FE budget,
accept ~0.08% of the seeds they produce, and produced zero best-solution updates at either
problem size.** Every improvement came from the mutation operators; branch B produced the final
Best at both sizes.

This is the paper's own design (Algorithm 2 lines 16–21) and it is not a defect in our port — the
branch frequencies match the pseudocode to within sampling noise (§4, Gate 4). It suggests a
concrete, testable improvement: **delete branches C/D and give their budget to A/B.** On this
evidence it cannot hurt. That is one flag away and is the most promising thing we found.

### 6.3 Population diversity collapses almost immediately

| | ta01 (D=225) | ta71 (D=2000) |
|---|---|---|
| distinct fitness values, start → end (of N=40) | 22 → **1** | 24 → **1** |
| mean pairwise distance, start → end | 47.2 → 27.1 | 136.5 → 73.6 |
| fraction of spread retained | 57.4% | 53.9% |

Within the first few percent of the budget, **all 40 trees carry the same fitness value** at both
sizes. The population becomes clones of the incumbent. This is the predicted consequence of
Algorithm 2 lines 7/10/11 seeding branches A/B from `Best`/`Tree_r` rather than from the parent
tree (the ATSA design notes) — the paper frames its contribution as "three mutation
operators", but the operative change may be "half the seeds are mutated copies of the incumbent".
Equally severe at both sizes; we report it as a property of the algorithm, not as an explanation
for anything.

## 7. Convergence — a falsifiable claim from the paper

The paper's §5 says of Figure 7: *"the ATSA approach converges slowly at the first stage but
continues to converge until the given number of FEs is completed."*

**(a) Monotonicity — passed.** A best-so-far curve can never rise. Across **84 runs**: **0 rises**.

**(b) The described shape holds.** Best-so-far averaged over seeds:

| inst | D | runs | 25% | 50% | 75% | 100% | Q4 gain | last improvement (mean % of budget) |
|---|---|---|---|---|---|---|---|---|
| ta01 | 225 | 20 | 1485.6 | 1475.1 | 1466.2 | 1460.3 | 5.9 | 42.4% |
| ta02 | 225 | 20 | 1460.2 | 1448.2 | 1445.8 | 1440.8 | 5.0 | 40.4% |
| ta03 | 225 | 20 | 1445.0 | 1439.8 | 1439.0 | 1433.1 | 6.0 | 38.8% |
| ta04 | 225 | 20 | 1446.2 | 1427.8 | 1422.3 | 1418.3 | 4.0 | 38.5% |
| ta71 | 2000 | 4 | 6543.5 | 6466.0 | 6456.8 | 6441.2 | 15.5 | 68.5% |

Every quarter still yields a gain on average, and **20–30% of D=225 runs and 50% of D=2000 runs
are still improving in the final quarter.** *"Until the FEs are completed"* overstates the typical
run — the mean last improvement lands at ~40% (D=225) and ~69% (D=2000) — but the claim is
directionally right and we do not contradict it.

> **A hypothesis we formed and killed.** A single seed-0 trace showed ta01 improving to 89% of
> budget and ta71 stopping at 29% — a textbook "large instances stagnate early" fingerprint.
> Across seeds it **reverses** (40.0% vs 68.5%; Mann–Whitney p = 0.95 against). One run was
> unrepresentative in both directions. Every convergence number above is a mean over seeds.

## 8. The TSA bisection — the baseline reproduces too

We implemented the paper's **Algorithm 1 (basic TSA)** as `src/atsa_jssp/tsa.py` and ran it under
identical conditions (N=40, ST=0.2, NS∈[4,10], MaxFEs=D×1000, seeds 0–19; 180 runs). TSA shares
everything with ATSA except the mutation operators — same decoder, RK encoding, Eq.3/Eq.4, FE
accounting, harness, seeds — so it independently exercises the shared core at both ends of the
size range. Its FE accounting is exact by construction: **measured E[FE/seed] = 1.0000**.

| inst | D | our TSA | paper TSA | diff% |
|---|---|---|---|---|
| ta01 | 225 | 1737.5 | 1724.55 | +0.75 |
| ta02 | 225 | 1728.8 | 1725.60 | +0.18 |
| ta03 | 225 | 1708.7 | 1736.70 | −1.62 |
| ta04 | 225 | 1736.6 | 1716.35 | +1.18 |
| ta05 | 225 | 1740.5 | 1736.60 | +0.22 |
| ta71 | 2000 | 8251.2 | 8371.05 | −1.43 |
| ta72 | 2000 | 7896.4 | 7984.40 | −1.10 |
| ta73 | 2000 | 8245.2 | 8326.20 | −0.97 |
| ta74 | 2000 | 8031.2 | 8076.15 | −0.56 |

**Both algorithms reproduce across the size range.** Every TSA instance is within **1.7%** of the
paper's TSA column, at both D=225 and D=2000.

And the quantity the paper actually claims — how much ATSA gains over TSA — reproduces too, with
each side computed within its own source so implementation differences cancel:

| | our ATSA gain over our TSA | paper's ATSA gain over paper's TSA |
|---|---|---|
| D=225 (5 instances) | **16.84%** | **16.70%** |
| D=2000 (4 instances) | **23.36%** | **24.12%** |
| ta71 specifically | **22.67%** | **23.06%** |

> Two honest notes. (i) At D=2000 our TSA is consistently ~1% *better* than the paper's (4/4
> instances, −0.56% to −1.43%). Because TSA's run-to-run variance is small (std ≈ 60 on a mean of
> ≈ 8000), that ~1% is statistically detectable (mean z = −6.2) even though it is a reproduction
> by any practical standard. (ii) Coverage is ta01–05 and ta71–74 at 20 runs each; ta75 was not
> run — every large instance measured agrees, so it is confirmatory. `uv run python
> run_tsa_bisect.py` will run only ta75.

## 9. How we caught a transcription error that looked like a discovery

**This section exists because the finding this report previously led with was an artefact of our
own data entry, and the method that caught it is more useful than the finding would have been.**

An earlier draft reported a **size-dependent drift**: our reproduction supposedly degraded as
problem size grew, with z rising monotonically from ≈0 at D=225 to **+9.1** at D=2000,
**25/25** instances with D≥450 worse than the paper, and **corr(D, diff%) = +0.72**. It was
statistically overwhelming, it was reproducible, and it landed exactly where the paper stakes its
claim. It was also entirely false.

**The cause:** the paper's ATSA reference column had been hand-transcribed into a Python dict and
was **wrong on 30 of 40 rows** — every instance from ta21 onward. Crucially the errors were
**systematic and grew with D** (ta21 off by 21, ta51 by 83, **ta71 by 237**), and they biased the
reference *downward* at large D, which made the paper look better and us look worse in exactly
the pattern of a size-dependent effect. A random transcription error would have produced noise
and been obvious. A *structured* one produced a clean, plausible, publishable-looking result.

**The check that caught it, which was available the whole time:** Table 5 prints its **own AVG
row**. It is an independent checksum on every column, and it takes one line to run.

| | column average | paper's AVG row | |
|---|---|---|---|
| Our transcribed ATSA mean column | **2793.38** | 2839.35 | **off by 46** |
| Corrected column (as transcribed here) | **2839.3675** | 2839.35 | agrees to **+0.0175** — see the residual note below |

The TSA column, transcribed at the same time, passed the same check — which is how the error was
localised to the ATSA column specifically.

**What changed as a result:**

- Table 5 is no longer transcribed by hand into working code. It was parsed once, offline, from
  the source paper into `src/atsa_jssp/paper_table5.py`, which carries a "do not hand-edit" banner.
  That committed file is the only copy of **Şahman (2022) Table 5** in this repository. (The DTSA
  paper's tables, Cinar 2020, live separately in `dtsa/dtsa_tables.py`.)
- `tests/test_paper_table5.py` asserts **all 20 columns** (5 algorithms × mean/med/min/max)
  against the paper's own AVG row, on every test run. The ATSA-mean column has its own named test
  quoting the 2793.38 failure, so a regression is unmissable.
- The drift finding is deleted. The corrected numbers are §5: corr = **−0.02**, no drift.

> **Residual, disclosed.** Recomputing all 20 columns against the printed AVG row: 19 reconcile to
> within **0.005**; the ATSA-mean column leaves a residual of **+0.0175** (it averages 2839.3675 vs
> the printed 2839.35 — its column sum is ~0.7 higher than the printed AVG implies). This is either
> the paper's own AVG rounding or a single ATSA cell off by ~0.7, and it cannot be resolved without
> the source PDF, which is not redistributed here. No value was adjusted to hide it; the checksum
> test tolerance is **0.02**, set just above this residual (not an exact-match guarantee).

**The lesson worth carrying to the group:** we nearly presented a data-entry mistake as the most
interesting result anyone had found, and the reason it survived internal scrutiny was that it was
*statistically strong* — z = 9 felt like evidence. Statistical strength measures consistency with
the reference; it says nothing about whether the reference is right. **Any transcribed reference
column needs a checksum before any comparison against it is believed.** When a published table
gives you its own AVG row, that is not decoration — it is the check the authors handed you.

## 10. Distance from best known — stated precisely

Metaheuristic quality has to be judged against the true optimum, not another metaheuristic.
**Most of these bounds are not proven.**

- **ta01's optimum is proven: 1231.** Our best over the 20-run protocol is **1382 (+12.3%)**. The
  paper's reported best is **1347 (+9.4%)** — also over 20 runs. Like-for-like, the paper's best
  ta01 solution beats ours by 2.9 points of gap.
- **ta22, ta23, ta25, ta32, ta33, ta34 and all of ta41–ta45 are NOT proven optimal** — they are
  bounded. Wherever a value is not proven, this report says **"best known"**, never "optimum".

OR-Tools CP-SAT solves ta01 to proven optimality (1231) in seconds. **ATSA is competitive against
basic random-key-encoded metaheuristics, not against the state of the art in job-shop
scheduling.** Anyone evaluating a scheduler for production use needs that stated plainly.

## 11. Scope limits and notes on the paper

**Scope limits**

- **ATSA and basic TSA only.** PSO, ABC and GWO were not assigned to us and were not implemented.
- **Therefore no Friedman test** — it requires ≥3 algorithms. We compare directly against the
  paper's published Table 5 instead, and have not borrowed anyone else's baselines to manufacture
  a third arm.
- No local search, no left-shifting, no parameter tuning. Table 4 exactly as printed.
- The TSA bisection covers 9 of the 40 instances (both ends of the size range), not all 40.

**Notes on the paper** — each verified against the source paper (cited above; PDF not redistributed).

| | |
|---|---|
| **Worked example** | §2's text prints `Cmax = 24` (verbatim: *"makespan (C_max) is 24 units of time"*). Figure 1's own title says `Makespan=22`, and 22 is what the schedule in Figure 1 produces. We used **22** as our unit test. |
| **Improvement denominator** | The headline 30.33% / 12.04% divide by the improved value, not the baseline. Non-standard, never stated, inflates every headline. Conventionally: 23.25% / 10.75%. See §6.1. |
| **ST direction** | §3 prose says *"if the st parameter is less than the randomly selected value … Eq.3 is used"* (ST < rand). Algorithm 1 line 10 says `if rand < ST`. Opposite. We followed the **pseudocode**. |
| **Hard-coded constants** | The seed mechanism hard-codes **0.5** and **0.75** where basic TSA uses `ST`. Never justified, never tuned, no sensitivity analysis. |
| **Symmetry block size** | The distribution of the block-size parameter `r2` is unspecified. |
| **Table 5 W/T row** | Reads **48/48** over a 40-instance test set. |
| **E[FE/seed]** (ours) | 1.4 holds only under the pseudocode reading of ST. Branch B is the only 2-FE branch, so E = 1 + P(B); under the prose reading it is **1.1**. |

## 12. Appendix — the 40 instances not in the paper

The Taillard distribution contains ta01–ta80; the paper uses 40 of them. We ran the other 40 (ta06–10,
ta16–20, ta26–30, ta36–40, ta46–50, ta56–60, ta66–70, ta76–80) under identical settings — 20 runs,
seeds 0–19. **The paper reports nothing for these, so no comparison is possible.** Reference data
for the group; no claim is made about them.

## 13. Open questions — measured, not guessed

- **`operator_anchor` (A3) is untested.** A new config flag switches branches A/B to seed from the
  parent tree instead of `Best`/`Tree_r`, as basic TSA does. **Default stays `literal`, exactly as
  the paper prints it.** With no drift left to explain, this is no longer a lead — it is an open
  question about a documented ambiguity, worth an ablation on its own merits (§6.3 shows the
  anchoring drives the diversity collapse).
- **Deleting branches C/D** and reallocating their 35.7% of budget to A/B (§6.2). Zero best-updates
  across 2M FEs says this should not hurt. This is the one genuine research lead we have.
- **The symmetry block-size distribution** is unspecified in the paper and is an ATSA-only degree
  of freedom.

---

### Reproducing this work

```bash
uv sync
uv run python scripts/verify_data.py                          # 80/80 across formats
uv run pytest -q                                              # all six gates, incl. the Table 5 checksum
uv run python -m atsa_jssp.cli run --instance ta01 --runs 20   # one instance
uv run python -m atsa_jssp.cli run --all --runs 20 --jobs 8    # the full paper set
```

Every result CSV carries the seed, final FE count, full configuration and the git SHA that
produced it.
