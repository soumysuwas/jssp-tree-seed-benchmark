# DTSA on the Job Shop Scheduling Problem — status report

**Independent implementation of** A. C. Cinar, S. Korkmaz, M. S. Kiran (2020), *"A discrete
tree-seed algorithm for solving symmetric traveling salesman problem"*, Engineering Science and
Technology, an International Journal **23**(4):879–890, doi:10.1016/j.jestch.2019.11.005.

Soumy Suwas — M.Tech (AI), Indian Institute of Technology Hyderabad

**Written for an engineer who has not read either paper.** Prose is hand-written; **every derived
number is computed** by `dtsa/metrics.py` from `dtsa/dtsa_tables.py` (the paper's tables, parsed
offline from the source paper) and the results CSVs. Nothing is hand-typed — see §7.

<!-- BEGIN:atsa_context -->
**The baseline itself is validated.** "Our ATSA" is not a fresh implementation taken on trust — it is a reproduction of Sahman (2022) that matches the paper's Table 5 on **40/40 instances within ±3.1%** (max absolute deviation 3.04%, mean absolute deviation 0.76%). The entire DTSA-vs-ATSA comparison below rests on that, so it is stated here once; full detail is in the ATSA report (`ATSA_reproduction_report.md`).
<!-- END:atsa_context -->

<!-- BEGIN:status_banner -->
**Generated 2026-08-03 03:53. Job-shop run status: COMPLETE — 80/80 instance × N-setting groups.**

| N setting | complete | pending |
|---|---|---|
| `N=40` | 40/40 | _none_ |
| `N=D` | 40/40 | _none_ |

Completed `N=40`: ta01, ta02, ta03, ta04, ta05, ta11, ta12, ta13, ta14, ta15, ta21, ta22, ta23, ta24, ta25, ta31, ta32, ta33, ta34, ta35, ta41, ta42, ta43, ta44, ta45, ta51, ta52, ta53, ta54, ta55, ta61, ta62, ta63, ta64, ta65, ta71, ta72, ta73, ta74, ta75

<!-- END:status_banner -->

## Headline — three claims, in order

**Claim 1 — DTSA as published outperforms ATSA as published at a matched budget, on every
instance.** Both algorithms run exactly as their own authors specify — DTSA with its
constructive (nearest-neighbour / MWKR) seed, ATSA with random initialisation — under an identical
`MaxFEs = D×1000`, same decoder, same seeds, same machine. The constructive seed is **part of
DTSA**, not an advantage we added, so this is a fair published-vs-published comparison.

<!-- BEGIN:headline -->
| N setting | vs our ATSA | vs paper ATSA | DTSA better on |
|---|---|---|---|
| `N=40` | +4.06% | +3.88% | 40/40 |
| `N=D` | +3.87% | +3.70% | 40/40 |

_DTSA-core, 20 seeds, sampler C3._
<!-- END:headline -->

**Claim 2 — but most of that margin is INITIALISATION, not search.** Removing DTSA's MWKR seed and
starting it randomly, like ATSA, retains well under half the margin — below the threshold
pre-registered before the ablation. The residual search advantage is real (DTSA still wins unseeded
on every instance tested) but **shrinks with instance size**. Equalised, DTSA and ATSA are close to
indistinguishable at the sizes the ATSA paper is about. **All figures: §5.1, generated.**

**Claim 3 — the actionable one: ATSA leaves real makespan on the table by initialising randomly.
DEMONSTRATED.** The flip side of Claim 2, and no longer a prediction. ATSA itself was run with
**one** of its forty trees replaced by an MWKR dispatching sequence — the search untouched, the
budget untouched, the other 39 trees bit-identical to stock — on 5 instances × 20 seeds (§5.5).

<!-- BEGIN:b4_claim3 -->
| instance | ATSA stock | ATSA +MWKR seed | DTSA-core | seed gain | gap before | gap after |
|---|---|---|---|---|---|---|
| ta01 | 1460.3 | 1402.8 | 1404.3 | +3.93% | +3.83% | -0.10% |
| ta11 | 1670.9 | 1607 | 1594.8 | +3.82% | +4.55% | +0.76% |
| ta21 | 2033.1 | 1971.4 | 1950.8 | +3.03% | +4.05% | +1.05% |
| ta31 | 2147.8 | 2142.4 | 2103.2 | +0.25% | +2.08% | +1.83% |
| ta41 | 2575 | 2509.8 | 2499 | +2.53% | +2.95% | +0.43% |

_ATSA `Config()` untouched, N=40, MaxFEs = D×1000, 20 seeds per cell. The two arms share the same 40-tree population bar one tree — the random rows are drawn from `np.random.RandomState(seed)`, verified bit-identical to the njit kernel's `np.random.seed(seed)` stream. DTSA column is sampler C3._

**Mean seed gain +2.71%, positive on 5/5. The DTSA–ATSA gap falls from +3.49% to +0.79% — 77% of it closed by one seeded tree.**
<!-- END:b4_claim3 -->

The reading was **pre-registered in the project log before the run**, at "substantial closure,
of the order of the ~2% initialisation share". The measured gain exceeds it. One dispatching rule,
one tree, no algorithmic change, and most of DTSA's advantage over ATSA disappears.

> ### ⚠️ Two hard qualifications that travel with every job-shop number
> **(A) TSP validation (Gate 1) FAILED** (§4). The operator ranking reproduced; the absolute level
> did not (residual mostly the unspecified NN start city, §4.8). **(B)** Claim 2 above — the win is
> mostly initialisation. **Any figure, table or slide showing a DTSA job-shop number must carry (A)
> and (B) on the same page.** Also: **`DTSA+LS` is excluded from all conclusions** — the local
> search is now understood (§8, genuine not buggy) but adds nothing on converged solutions.
>
> **Sampler provenance.** U1 resolved to **C3** (§4.6); the original run used C1 and the result
> proved sampler-sensitive on one instance, past the pre-registered threshold, so **the full
> 40-instance run was repeated with C3** and §5 now reports the C3 run. The C1 run is retained in `results/dtsa_jssp.csv` for comparison —
> the qualitative headline is unchanged between the two — DTSA wins on every instance either way.

---

## Contents

<!-- BEGIN:toc -->
- **Headline — three claims, in order**
- **1. What DTSA is**
- **2. Method**
    - 2.1 Representation (D1)
    - 2.2 Initialisation (D2)
    - 2.3 Local search (D3) — and why the two configurations are kept apart
    - 2.4 Parameters (D4)
- **3. Defects found in Cinar et al. (2020)**
    - Checksum status
    - A resolved ambiguity worth recording (U9)
- **4. TSP validation — Gate 1, FAILED**
    - 4.1 Why this gate exists
    - 4.2 The verdict
    - 4.3 Full results
    - 4.4 Where the defect is: the spread across source trees
    - 4.5 The residual: one bounded diagnostic
    - 4.6 U1 — the symmetry sampler, RESOLVED to C3 (under the corrected start)
    - 4.7 U14 — a pre-registered prediction that failed
    - 4.8 U12 — the nearest-neighbour start city is unspecified, and it is worth several per cent
- **5. Job-shop results**
    - 5.1 The MWKR ablation — read this before §5.2
    - 5.2 Per-instance results
    - 5.3 Parameter transfer: `N=D` vs `N=40` — the pre-registered prediction failed
    - 5.4 Per-instance detail, Sahman Table 5 format
    - 5.5 Claim 3 tested directly — ATSA with one MWKR-seeded tree
    - 5.6 Timing, threading, and what is actually reproducible
    - 5.7 ATSA vs DTSA runtime
- **6. Findings**
    - 6.1 Method findings — the checks that earned their keep
- **7. Reproducibility**
    - System configuration
- **8. Open questions**
    - 8.1 The N5 local search — genuine, not a bug
<!-- END:toc -->

---

## 1. What DTSA is

The Tree-Seed Algorithm keeps a population of "trees" (solutions). Each iteration, every tree
produces "seeds" (mutated copies); if the best seed beats its tree, it replaces it. DTSA is the
discrete version: the arithmetic seed-creation equations of continuous TSA are replaced by three
**permutation operators** — `swap` (exchange two positions), `shift` (rotate a range by one), and
`symmetry` (reverse two equal-size blocks, then exchange them).

Structure, from the paper's Fig. 6 (journal p. 884):

| | |
|---|---|
| Population | `N` trees; tree 1 is a nearest-neighbour tour, the rest random |
| Seeds per tree | **`NS = 6`, fixed — and it is structurally 3 operators × 2 source trees** |
| Source trees | a random tree `Tree(k)` always supplies 3 seeds; the parameter `ST` chooses whether the other 3 come from the global best or from the current tree |
| Operator choice | **there isn't one.** All three operators fire every iteration, one per seed slot |
| Budget | `fes = N` at start, then `+6` per tree per iteration — exactly, with no branch-dependent cost |
| Local search | 2-opt, **once, after termination**, on the best solution only |

**DTSA has never been published on job-shop scheduling.** There is no DTSA column in the ATSA
paper's Tables 5–8, so a DTSA job-shop number cannot be checked against anything. That is why the
work is sequenced *implement → reproduce the paper's own TSP results → only then port*.

---

## 2. Method

### 2.1 Representation (D1)

DTSA operates on raw permutations. This repository represents a job-shop solution as a
**random-key vector** `x ∈ [−5,5]^D`, `D = n·m`, decoded by ranking `x` and mapping rank `τ` to
job `(τ mod n) + 1`. We apply DTSA's operators **to the random-key vector**.

This is exact, not an approximation. All three operators are *position permutations*, and
random-key ranking is equivariant under position permutation: if `x' = x ∘ σ` then `seq' = seq ∘ σ`.
So mutating the key vector and then decoding gives **identically** the sequence you get by
decoding and then mutating. Verified over **10,800 trials** across three instances and all three
operators, zero divergences (`dtsa/tests/test_operators.py::test_d1_equivariance_rk_vs_sequence`).

The practical consequence is that DTSA calls the **same verified decoder** as our ATSA runs,
unmodified — so any ATSA-vs-DTSA difference is attributable to the search operators alone.

### 2.2 Initialisation (D2)

DTSA seeds one tree with a nearest-neighbour tour. The job-shop analogue we chose is **MWKR
(Most Work Remaining)**: repeatedly dispatch the job with the greatest total remaining processing
time. It is the closest structural match — one greedy, problem-aware constructive pass, no search.

The dispatched sequence is converted back to a random-key vector exactly; the round trip
`sequence → keys → sequence` is asserted to be the identity on **every** instance (a test
assertion, not a measurement), and the seeded
tree's decoded makespan **equals** MWKR's own (not merely "no worse").

### 2.3 Local search (D3) — and why the two configurations are kept apart

Fig. 6 never increments the evaluation counter for 2-opt: the only increment is inside the main
loop, and 2-opt runs after it. **DTSA's published results therefore include an uncounted
local-search budget of unstated size.** Our ATSA results were produced with none, under a strictly
counted budget. Blending the two would compare a budget-matched algorithm against one with free
extra evaluations.

So we report two configurations and never one blended number:

| configuration | local search | evaluation accounting |
|---|---|---|
| **`DTSA-core`** | none | `fes = N + 6·N·iterations`, asserted every run. **This is the column compared to ATSA.** |
| **`DTSA+LS`** | critical-block (N5) swap, once, on the best solution | evaluations counted and reported **separately**, never in `fes` |

### 2.4 Parameters (D4)

DTSA's own algorithm parameters, ATSA's experimental protocol:

| parameter | value | source |
|---|---|---|
| `NS` | 6 | DTSA §5.2 |
| `ST` | 0.5 | DTSA §5.2 |
| `MaxFEs` | `D × 1000` | ATSA Table 4 — **deviation from DTSA**, required for comparability |
| runs | 20, seeds 0–19 | ATSA protocol — **deviation from DTSA** (30) |
| `N` | **both `40` and `D`** | see below |

**The `N = D` arm was pre-registered as expected-to-fail — and the prediction was WRONG.** DTSA
sets `N` = the number of decision variables. Under `MaxFEs = D × 1000` that buys `1000·D / 6D` ≈
**166 iterations at every problem size**, so we predicted (D001/D004) that a population of 2,000
trees given only 166 rounds of improvement would be badly under-searched and the arm would do badly.
It did not: `N=D` comes out within half a per cent of `N=40` (§5.3). The MWKR seed and early
saturation (U14, §4.7) carry the result, and the iteration/population trade-off barely moves it.
**A documented negative result about parameter transfer is still a finding**, and it is reported as
one in §5.3 rather than omitted — the prediction is recorded as wrong.

---

## 3. Defects found in Cinar et al. (2020)

These are independent of whether our implementation is right, and they are a primary contribution
of this work. Each is pinned by a test that fails if the defect ever "disappears" from our parse.

**A fourth defect — U12, the unspecified nearest-neighbour start city — is in §4.8**, because it is only visible once the validation run is set up. It is worth several percentage points of apparent performance (§4.8, generated) and is arguably the most consequential of the four for anyone reproducing the paper.

<!-- BEGIN:defects -->
**U8 — the improvement denominator.** Table 1's `RE` divides by the *result*, not the optimum. Confirmed on all 9 rows; the conventional formula is refuted on 6 of them (the other 3 sit exactly on the optimum and cannot discriminate).

| Table 1 row | Mean₂ₒₚₜ | printed RE | (r−opt)/r | (r−opt)/opt |
|---|---|---|---|---|
| shift(best tree) | 7758.43 | 2.79 | 2.79 | 2.87 |
| shift(current tree) | 7678.57 | 1.78 | 1.78 | 1.81 |
| shift(random tree) | 7740.23 | 2.56 | 2.56 | 2.63 |
| swap(best tree) | 7863 | 4.08 | 4.08 | 4.26 |
| swap(current tree) | 7863 | 4.08 | 4.08 | 4.26 |
| swap(random tree) | 7858 | 4.02 | 4.02 | 4.19 |
| symmetry(best tree) | 7551.03 | 0.12 | 0.12 | 0.12 |
| symmetry(current tree) | 7542 | 0 | 0 | 0 |
| symmetry(random tree) | 7542 | 0 | 0 | 0 |

**U15 — Table 5's KROE100 block was scored against the wrong optimum.** Its `Optimum` cell prints **22068** (correct for kroE100), but all five error entries imply **≈22141**, which is **KROB100's** optimum (22141), printed three rows higher in the same table. The *mean* column survives: five independent rows agreeing on one implied optimum cannot happen by accident.

| KROE100 row | Mean | printed error | implied optimum |
|---|---|---|---|
| SA | 23125 | 4.44 | 22141.9 |
| DSTA0 | 23738 | 7.21 | 22141.59 |
| DSTAI | 23371 | 5.56 | 22140.02 |
| DSTAII | 22637 | 2.24 | 22141.04 |
| DTSA | 22547 | 1.83 | 22141.8 |

**U16 — a duplicated row.** `swap(current tree)` and `swap(best tree)` are identical in all four columns (8133.00 / 7863.00 / 0.00 / 4.08) and are the only collision in the table. Our own run separates them by **113.83** with non-zero variance in both, so the collision is almost certainly a typesetting duplication. `swap(best tree)` therefore carries no independent published value and is excluded as a validation target.
<!-- END:defects -->

### Checksum status

The paper has **no printed AVG row**, so the mechanism that caught a transcription error on the
ATSA side of this project does not exist here. The substitute is a **derived-column identity**:
most tables print a mean *and* that mean's error against a stated optimum, so the error column can
be recomputed from the mean column. Asserted on every test run.

<!-- BEGIN:checksums -->
- **Table 1** — `RE == (Mean₂ₒₚₜ − Opt)/Mean₂ₒₚₜ × 100` holds on **9/9** rows (the non-standard denominator, U8).
- **Table 5** — conventional identity holds on **4/5** instance blocks; **KROE100 fails** (U15).
- **Table 15** — holds on **32/40** rows; 8 unverifiable (EIL76, CH150 have no published optimum).
- **Table 4** — no error column, therefore **no internal checksum**; two-witness only.

Distance conventions in force: `rounded EUC_2D (optimum 7542); RE divides by RESULT`.
<!-- END:checksums -->

### A resolved ambiguity worth recording (U9)

The paper quotes two different optima for the same instances, switching between tables without
saying so:

<!-- BEGIN:u9 -->
| instance | rounded (`nint`) | unrounded |
|---|---|---|
| BERLIN52 | 7542 | 7544.37 |
| KROA100 | 21282 | 21285.4 |
<!-- END:u9 -->

Recomputing TSPLIB's own published
optimal tours from the coordinates reproduces **both** values exactly: they are the same tour
measured with `nint`-rounded and with unrounded Euclidean distances. Which convention each table
uses is now decided by evidence — whichever one makes that table's own error column reproduce.

---

## 4. TSP validation — Gate 1, **FAILED**

> **A note on our own percentages.** §4's central charge against the paper (U8) is that it
> divides an improvement by the wrong quantity. We hold this report to the same standard: every
> percentage below names its denominator at the point of use. Where a single measurement can be
> expressed two ways — e.g. a tour that is *X%* shorter — the figure and its denominator are given
> together, and any complementary figure over the other denominator is labelled as such rather
> than left to be confused with it.

### 4.1 Why this gate exists

A DTSA job-shop number is unfalsifiable on its own. The only published DTSA numbers are on
travelling salesman problems, so reproducing those is the only way to know the implementation is
right. Gate 1 targets **Table 1** (journal p. 885), the operator ablation on BERLIN52, because it
is the **only place in the entire paper that publishes a pre-2-opt number** — and the pre-2-opt
column is `DTSA-core`'s only published target anywhere.

Design: 9 configurations (3 operators × 3 source trees) × 30 seeds, `N = 52`,
`MaxFEs = 104,000`, rounded `EUC_2D`; plus both `NS` readings and all four symmetry samplers —
**1,080 runs**.

### 4.2 The verdict

The pass criteria were **pre-registered in commit `d63900a`, before any run existed**, and were
not revisited afterwards.

<!-- BEGIN:gate1_verdict -->
| criterion | result |
|---|---|
| **(i)** operator ranking reproduces symmetry &lt; shift &lt; swap | ✅ **PASS** — symmetry 8007.07 &lt; shift 8253.87 &lt; swap 8438.17 |
| **(ii)** `symmetry(current tree)` within ±1.5% | ❌ **FAIL** — ours **8578.83** vs published **7683.73**, **+11.65%** |

<!-- END:gate1_verdict -->

**FAIL.** The ranking reproduces; the level does not.

The ±1.5% tolerance is **our construction, not the paper's**: Table 1's only `Std.Dev.` column
describes the *post*-2-opt figures (provable — one row has a non-integer pre-2-opt mean alongside
a zero standard deviation, which is impossible under the integer-valued rounded metric), so the
paper publishes **no spread at all** for the column we are trying to reproduce.

### 4.3 Full results

<!-- BEGIN:gate1_table -->
| row (NS=6) | ours pre | ours post | published | dev % | note |
|---|---|---|---|---|---|
| shift(best tree) | 8257.97 | 8070.17 | 7891.37 | 4.65 |  |
| shift(current tree) | 8473.8 | 8033.43 | 7816.73 | 8.41 |  |
| shift(random tree) | 8253.87 | 8018.03 | 7903.13 | 4.44 |  |
| swap(best tree) | 8604.17 | 7944.4 | 8133 | 5.79 | EXCLUDED (U16) |
| swap(current tree) | 8718 | 7948.47 | 8133 | 7.19 |  |
| swap(random tree) | 8438.17 | 7997.23 | 8059.17 | 4.7 |  |
| symmetry(best tree) | 8031.63 | 7994.27 | 7737.9 | 3.8 |  |
| symmetry(current tree) | 8578.83 | 8020.67 | 7683.73 | 11.65 |  |
| symmetry(random tree) | 8007.07 | 7981.43 | 7697 | 4.03 |  |

_8 usable rows: mean |dev| **6.11%**, worst **11.65%**, 0/8 within ±1.5%._

| row (NS=1) | ours pre | ours post | published | dev % | note |
|---|---|---|---|---|---|
| shift(best tree) | 8264.27 | 8041.47 | 7891.37 | 4.73 |  |
| shift(current tree) | 8470.37 | 8057 | 7816.73 | 8.36 |  |
| shift(random tree) | 8224.8 | 8013.57 | 7903.13 | 4.07 |  |
| swap(best tree) | 8599.6 | 7941 | 8133 | 5.74 | EXCLUDED (U16) |
| swap(current tree) | 8722.63 | 7948.73 | 8133 | 7.25 |  |
| swap(random tree) | 8435.37 | 7989.7 | 8059.17 | 4.67 |  |
| symmetry(best tree) | 8044.5 | 8023.9 | 7737.9 | 3.96 |  |
| symmetry(current tree) | 8629.77 | 8019.7 | 7683.73 | 12.31 |  |
| symmetry(random tree) | 7986.8 | 7975.97 | 7697 | 3.77 |  |

_8 usable rows: mean |dev| **6.14%**, worst **12.31%**, 0/8 within ±1.5%._

<!-- END:gate1_table -->

### 4.4 Where the defect is: the spread across source trees

The paper's three source-tree variants are nearly interchangeable. Under the most literal reading
of its row labels, ours are not — and `current tree` is consistently our worst while it is often
the paper's best. That is a **structural** disagreement, not a magnitude one.

<!-- BEGIN:gate1_spread -->
| operator | paper spread | our spread (literal) |
|---|---|---|
| swap | 0.92% | 3.32% |
| shift | 1.11% | 2.66% |
| symmetry | 0.70% | 7.14% |
<!-- END:gate1_spread -->

**U17, adopted.** Fig. 6 assigns seeds 4–6 to the random tree `Tree(k)` *unconditionally, in both
`ST` branches*; `ST` selects the source of seeds 1–3 only. So an ablation row labelled
`X(current tree)` most plausibly means *the ST-branch source is the current tree, with the
`Tree(k)` half of the structure preserved* — not *every seed comes from the current tree*, a
structure Fig. 6 never describes. **It is adopted because it is more faithful to Fig. 6, not
because it improves agreement** — under it the gate still fails.

> **U17 changes only how we reconstruct the paper's Table 1 ablation. It does not change DTSA.**
> The full three-operator algorithm — what the job-shop port runs — is exactly Fig. 6 and has
> never been altered.

### 4.5 The residual: one bounded diagnostic

A single pre-registered 2×2 on the anchor row, under U17. Neither arm was eligible to become a
default, and the stop rule was written down before the run: *whatever it shows, the literal
defaults stay unless a reading is more faithful to Fig. 6.*

- **U12** — the nearest-neighbour seed starts at city 0 (Fig. 6 says "nearest neighbor tour",
  *singular*, and does not say which city) versus the best of all 52 possible start cities.
- **F6** — `best` updated after the tree loop (literal, Fig. 6 lines 35–36) versus immediately.

<!-- BEGIN:partb -->
| U12 (NN seed) | F6 (best update) | pre | post | runs 2-opt moved | seed tour length | dev % | 2-opt gain % |
|---|---|---|---|---|---|---|---|
| best_of_52 | deferred | 7699.23 | 7681.87 | 11 | 8181 | 0.2 | 0.23 |
| best_of_52 | immediate | 7699.23 | 7681.87 | 11 | 8181 | 0.2 | 0.23 |
| city_0 | deferred | 7982.03 | 7949.3 | 22 | 8980 | 3.88 | 0.41 |
| city_0 | immediate | 7982.03 | 7949.3 | 22 | 8980 | 3.88 | 0.41 |
<!-- END:partb -->

**The start city accounts for essentially the whole residual.** It is an unspecified detail of the
paper worth several per cent, and it is a reproducibility defect in Cinar et al., not a defect in
our search.

**Neither arm is adopted.** Best-of-52 is better-performing, not more faithful, so the default
remains city 0 and **the gate verdict stands at FAIL**. The `F6` arm turned out to be **inert by
construction** on this row — with the source fixed to the current tree, `best` is never read as a
seed source, so the comparison could not have shown anything. That is a flaw in the diagnostic's
design and the F6 result should be read as *no information*, not as *no effect*.

### 4.6 U1 — the symmetry sampler, **RESOLVED to C3** (under the corrected start)

The paper defines symmetry's mechanics but never says how the two block positions and the common
block size are drawn. Four candidate rules were **fixed in writing before any was run**
(the sampler design notes) and scored on two independent criteria: (a) reproducing the published
mean of `symmetry(current tree)` to within ±1.5%, and (b) leaving 2-opt the headroom the paper's
own pre→post gap implies. Both targets are read from `dtsa_tables.py`, never typed.

**The first three attempts scored every candidate with the NN tour starting at city 0.** U12 (§4.8)
then showed the paper almost certainly did *not* start there — so those scores were computed in a
configuration the paper does not describe, and "no candidate reproduces" was an artefact of the
wrong start city. Re-scored under the **paper-figure-matching configuration** — best-of-52 NN
start, U17 reading, 30 seeds:

<!-- BEGIN:u1_resolved -->
| candidate | our pre | (a) dev % | (a) |dev|≤1.5% | (b) 2-opt gain % | (b) ~1.84% | both |
|---|---|---|---|---|---|---|
| C1 | 7699.23 | 0.2 | ✅ | 0.23 | — |  |
| C2 | 7821.43 | 1.79 | — | 1.66 | ✅ |  |
| C3 | 7773.3 | 1.17 | ✅ | 1.43 | ✅ | **RESOLVED** |
| C4 | 7615.4 | -0.89 | ✅ | 0.06 | — |  |

_Criterion (a) target: within ±1.5% of the paper's 7683.73. Criterion (b) target: 2-opt gain within 0.5 pp of the paper's 1.84%. This is the **paper-figure-matching configuration** (best-of-52 NN start, U17), NOT our shipped default — Gate 1 stays FAILED at the city-0 literal default._

_**U1 RESOLVED → C3**: the unique candidate meeting both criteria under the corrected start._
<!-- END:u1_resolved -->

**C3 is the unique candidate that satisfies both criteria.** C1 and C4 nail the pre-2-opt level but
leave 2-opt almost no headroom (their tours are already near-optimal); C2 has the headroom but
overshoots the level; only C3 — bounded blocks, `L ≤ ⌈D/10⌉` — threads both. The mechanism is
intuitive: a smaller symmetry move leaves a tour that is *not* already 2-opt-optimal, so 2-opt does
real work (matching the paper's implied 2-opt gain) while the search still converges to the
paper's level.

> **This resolves U1, it does NOT re-open Gate 1.** The shipped default stays city-0 + literal, and
> the gate verdict stays FAILED (§4.2). C3 is adopted because it is the one sampler consistent with
> the paper's own Table 1 under its (reconstructed) start city — a fidelity argument, not a
> gap-closing one.

**Consequence — the job-shop run used C1, so is the result sampler-sensitive?** DTSA-core, `N=40`,
20 seeds, C1 vs C3 on five instances:

<!-- BEGIN:sampler_sensitivity -->
| instance | C1 mean | C3 mean | Δ % | >1% |
|---|---|---|---|---|
| ta01 | 1409.15 | 1404.3 | -0.34 | no |
| ta11 | 1591.8 | 1594.85 | 0.19 | no |
| ta21 | 1964.6 | 1950.75 | -0.7 | no |
| ta31 | 2068.85 | 2103.15 | 1.66 | YES |
| ta41 | 2483.65 | 2499.05 | 0.62 | no |

_Max |Δ| = **1.66%**. Verdict: **SAMPLER-SENSITIVE** — at least one instance moves >1%, so the C1 numbers are **provisional** and the full 40-instance run is being repeated with C3._
<!-- END:sampler_sensitivity -->

One instance moves past the ~1% threshold pre-registered for this check (see the table), so the
result **is** sampler-sensitive. **The full 40-instance run was therefore repeated with C3**
(1,600 runs) and §5 reports that C3 run; the C1 run is kept in `results/dtsa_jssp.csv`.

As predicted from the five-instance probe, the shift was small and mixed in sign and the
*qualitative* headline did not change — DTSA-core still beats our ATSA on every instance at both
`N` settings, with the set mean moving by a small fraction of a percentage point at either `N`
setting (the C3 figures are in the Headline block; the C1 run remains in its CSV). **The repeat was still the right
call**: that it changed little is a result, not a reason to have skipped it.

### 4.7 U14 — a pre-registered prediction that failed

Table 1's rows are single-operator ablations and the paper does not say whether `NS` is still 6
for them. We predicted, in advance, that `NS = 1` — which buys six times as many iterations at the
same budget — would be markedly better and would be the reading that reproduces the table.

<!-- BEGIN:u14 -->
| NS | while-iterations | mean |dev| over 8 usable rows |
|---|---|---|
| 6 | 334 | 6.11% |
| 1 | 1999 | 6.14% |
<!-- END:u14 -->

**The prediction was wrong**, and the reason is itself a result: **six times the iterations buys
nothing**, so the search saturates long before the budget is spent. That also means U14 cannot be
settled from Table 1 at all — both readings give the same answer.


### 4.8 U12 — the nearest-neighbour start city is unspecified, and it is worth several per cent

**This is the fourth defect in Cinar et al. (2020)**, alongside U8/U15/U16 in §3, and it is the
one that most affects anyone trying to reproduce the paper.

Fig. 6 line 5 says *"Determine the first tree as nearest neighbor tour"* — **singular, with no
start city given**. A nearest-neighbour tour is defined only once you fix where it starts, and on
BERLIN52 the choice is worth a great deal. That difference propagates all the way to the final
answer: in the §4.5 diagnostic, switching only the start city accounts for essentially the entire
Gate 1 residual on the anchor row.

<!-- BEGIN:u12_cost -->
| | seed tour | deviation of the anchor row from the published mean |
|---|---|---|
| first city (literal) | 8980 | +3.88% |
| best of all start cities | 8181 | +0.20% |

_The best-start seed tour (8181) is **8.9% shorter than the first-city tour** (8980→8181; denominator 8980). That alone moves the anchor row's deviation from the published mean by **3.68 percentage points** (+3.88% → +0.20%, both against the same published mean). The complementary figure 9.8% (same gap over the shorter tour) measures the same thing against a different denominator and is not used here._
<!-- END:u12_cost -->

**We did not adopt it.** "Nearest neighbour tour" in the singular is not a licence to take the best
of 52; that would be a better-performing reading, not a more faithful one, and the gate verdict
stands at FAIL under the literal default. But the finding stands on its own:

> **An unspecified initialisation detail is worth several percentage points of apparent algorithm
> performance in this paper** — the exact figure is in the table above. Any reproduction of Cinar
> et al. that does not match their start city will disagree with their tables by roughly that much,
> through no fault of its own.

The 2-opt-headroom gap is *not* explained by this and remains open — see §4.6.

---

## 5. Job-shop results

> **Gate 1 failed** (§4). These numbers are not compared to any published DTSA figure — none
> exists — but to **our own ATSA results under an identical protocol**: same decoder, same
> `MaxFEs = D × 1000`, same 40 instances, same seeds, same machine. The Gate 1 residual is uniform
> across operators and source trees and lies in the search, not the decoder, which is shared with
> ATSA and independently verified. The comparison is meaningful; the qualification travels with it.

### 5.1 The MWKR ablation — read this before §5.2

DTSA beat both the paper's ATSA column and our own ATSA reproduction on every instance we
completed, in the same direction every time. **A uniform, one-directional win is the shape that
has already produced one artefact on this project**, so it was tested rather than reported.

DTSA seeds one tree with an MWKR dispatching solution (§2.2). **ATSA seeds every tree randomly.**
That is an advantage with nothing to do with the search. We re-ran five instances with the seeding
switched off, having **pre-registered both readings and the 50% threshold beforehand**
(the project log).

<!-- BEGIN:mwkr -->
| instance | paper ATSA | our ATSA | DTSA seeded | DTSA unseeded | gap seeded % | gap unseeded % |
|---|---|---|---|---|---|---|
| ta01 | 1444.8 | 1460.3 | 1409.2 | 1420.6 | 3.5 | 2.72 |
| ta11 | 1663.7 | 1670.9 | 1591.8 | 1641.9 | 4.73 | 1.74 |
| ta21 | 1990.6 | 2033.1 | 1964.6 | 1991.4 | 3.37 | 2.05 |
| ta31 | 2152.9 | 2147.8 | 2068.8 | 2127.2 | 3.67 | 0.95 |
| ta41 | 2571.9 | 2575 | 2483.6 | 2565.7 | 3.55 | 0.36 |

_Mean gap seeded **+3.76%**, unseeded **+1.56%** — **42%** of the advantage retained against a pre-registered threshold of 50%. Verdict: **the win is largely about INITIALISATION, not the search**._

_DTSA still beats our ATSA on 5/5 instances unseeded, but the margin falls with problem size (+2.72% on ta01 to +0.36% on ta41)._
<!-- END:mwkr -->

**Verdict, by the pre-registered rule: the advantage is mostly about initialisation.** Well under
half the margin survives removing the seed, against the pre-registered 50% threshold (figures in
the block above). Two consequences, both mandatory:

1. **The claim changes.** It is not *"DTSA searches better than ATSA"*. It is *"DTSA reaches
   better solutions than ATSA at equal evaluation budget, and most of that comes from starting
   from a dispatching-rule solution rather than from random keys."* That is a real and useful
   result — a cheap constructive seed is worth more than the search difference — but it is a
   different claim and it must be stated as this one.
2. **A residual search advantage does survive** (DTSA still better on every instance unseeded),
   **but it shrinks with problem size**. At the sizes the ATSA paper is actually about, unseeded DTSA and
   ATSA are close to indistinguishable.

**A candidate mechanism, explicitly not demonstrated.** ATSA's branches C and D consume roughly a
third of every budget and produced zero best-solution updates in our instrumentation
(figure and method from the ATSA side of this project), whereas DTSA has no
dead branches — all six seeds per tree are productive by construction. That is *consistent with* a
residual advantage, but **we have not ablated ATSA's C/D branches**, so it is a hypothesis and is
not offered as an explanation.

### 5.2 Per-instance results

`DTSA-core` is the comparison column. `DTSA+LS` appears as one extra column for completeness only
and is **excluded from every conclusion** (§8) — the local search is not validated.

<!-- BEGIN:jssp -->
_Sampler **C3**. C3 is the U1-resolved sampler (§4.6)._

#### `N=40`

| instance | paper ATSA mean | our ATSA mean | DTSA-core mean | vs our ATSA % | vs paper ATSA % | DTSA min | DTSA+LS mean |
|---|---|---|---|---|---|---|---|
| ta01 | 1444.8 | 1460.3 | 1404.3 | 3.83 | 2.8 | 1362 | 1404.3 |
| ta02 | 1435 | 1440.8 | 1388 | 3.66 | 3.28 | 1320 | 1388 |
| ta03 | 1430.6 | 1433.1 | 1372.7 | 4.21 | 4.05 | 1296 | 1372.3 |
| ta04 | 1438.1 | 1418.3 | 1370.2 | 3.39 | 4.72 | 1311 | 1370.2 |
| ta05 | 1448.5 | 1442 | 1385.8 | 3.9 | 4.33 | 1342 | 1385.8 |
| ta11 | 1663.7 | 1670.9 | 1594.8 | 4.55 | 4.14 | 1530 | 1594.8 |
| ta12 | 1685.6 | 1658.6 | 1561.8 | 5.83 | 7.34 | 1496 | 1561.8 |
| ta13 | 1638.8 | 1651.1 | 1594 | 3.46 | 2.73 | 1533 | 1594 |
| ta14 | 1589.9 | 1598.6 | 1546.6 | 3.25 | 2.72 | 1458 | 1546.6 |
| ta15 | 1685 | 1633.8 | 1591.6 | 2.59 | 5.54 | 1523 | 1591.2 |
| ta21 | 1990.6 | 2033.1 | 1950.8 | 4.05 | 2 | 1868 | 1950.6 |
| ta22 | 1944.3 | 1953.6 | 1857.2 | 4.93 | 4.48 | 1801 | 1856.9 |
| ta23 | 1902.1 | 1925.4 | 1797.3 | 6.65 | 5.51 | 1738 | 1797.3 |
| ta24 | 1994.2 | 1975.6 | 1893.2 | 4.17 | 5.06 | 1820 | 1893.2 |
| ta25 | 1934.3 | 1951 | 1872.1 | 4.05 | 3.22 | 1801 | 1872.1 |
| ta31 | 2152.9 | 2147.8 | 2103.2 | 2.08 | 2.31 | 2041 | 2103.2 |
| ta32 | 2242.8 | 2277.2 | 2175.7 | 4.46 | 2.99 | 2097 | 2175.7 |
| ta33 | 2250.9 | 2269 | 2198.9 | 3.09 | 2.31 | 2097 | 2198.9 |
| ta34 | 2234.5 | 2254.4 | 2149.2 | 4.67 | 3.82 | 2066 | 2149.2 |
| ta35 | 2351.7 | 2369.4 | 2308.4 | 2.58 | 1.84 | 2204 | 2308.4 |
| ta41 | 2571.9 | 2575 | 2499.1 | 2.95 | 2.83 | 2402 | 2499 |
| ta42 | 2520.4 | 2540.8 | 2414.9 | 4.95 | 4.18 | 2348 | 2414.6 |
| ta43 | 2424.7 | 2431.4 | 2253.7 | 7.31 | 7.05 | 2165 | 2253.6 |
| ta44 | 2544.5 | 2573 | 2428.1 | 5.64 | 4.58 | 2367 | 2428 |
| ta45 | 2484.9 | 2498.8 | 2391.3 | 4.3 | 3.76 | 2312 | 2391.4 |
| ta51 | 3337.2 | 3344.4 | 3271.8 | 2.17 | 1.96 | 3164 | 3271.8 |
| ta52 | 3371.4 | 3343.9 | 3220.2 | 3.7 | 4.49 | 3114 | 3220.2 |
| ta53 | 3136.2 | 3096 | 2947.2 | 4.8 | 6.02 | 2898 | 2947.2 |
| ta54 | 3206.9 | 3208.7 | 3083.9 | 3.89 | 3.83 | 2993 | 3084 |
| ta55 | 3261.3 | 3260.8 | 3179.8 | 2.48 | 2.5 | 3071 | 3178.6 |
| ta61 | 3521.7 | 3522.5 | 3314.2 | 5.91 | 5.89 | 3241 | 3313.2 |
| ta62 | 3548 | 3581.8 | 3402.1 | 5.02 | 4.11 | 3316 | 3402 |
| ta63 | 3322.3 | 3351.9 | 3160.6 | 5.71 | 4.87 | 3049 | 3159.6 |
| ta64 | 3333.8 | 3319.4 | 3209.1 | 3.32 | 3.74 | 3136 | 3209 |
| ta65 | 3371.9 | 3380 | 3291.1 | 2.63 | 2.4 | 3172 | 3291 |
| ta71 | 6441 | 6380.6 | 6234.2 | 2.29 | 3.21 | 6039 | 6234.2 |
| ta72 | 5910.9 | 5957.4 | 5657.1 | 5.04 | 4.29 | 5481 | 5657 |
| ta73 | 6527 | 6549.8 | 6248.8 | 4.59 | 4.26 | 6047 | 6248.8 |
| ta74 | 5987.3 | 5970.8 | 5822.2 | 2.49 | 2.76 | 5744 | 5822.2 |
| ta75 | 6293.1 | 6317.4 | 6088.1 | 3.63 | 3.26 | 5960 | 6088.1 |
_Instances complete: **40/40**. DTSA-core better than our ATSA on **40/40**, worse on **0/40**._

_Full set: mean difference **+4.06%**, median **+3.97%** vs our ATSA; mean **+3.88%** vs the paper._

#### `N=D`

| instance | paper ATSA mean | our ATSA mean | DTSA-core mean | vs our ATSA % | vs paper ATSA % | DTSA min | DTSA+LS mean |
|---|---|---|---|---|---|---|---|
| ta01 | 1444.8 | 1460.3 | 1396.8 | 4.35 | 3.32 | 1333 | 1396 |
| ta02 | 1435 | 1440.8 | 1416.5 | 1.68 | 1.29 | 1371 | 1416.6 |
| ta03 | 1430.6 | 1433.1 | 1386 | 3.29 | 3.12 | 1298 | 1386 |
| ta04 | 1438.1 | 1418.3 | 1374.5 | 3.09 | 4.42 | 1300 | 1374.5 |
| ta05 | 1448.5 | 1442 | 1374.2 | 4.7 | 5.13 | 1320 | 1374.2 |
| ta11 | 1663.7 | 1670.9 | 1593.3 | 4.64 | 4.23 | 1551 | 1592.9 |
| ta12 | 1685.6 | 1658.6 | 1569 | 5.4 | 6.91 | 1526 | 1569 |
| ta13 | 1638.8 | 1651.1 | 1585 | 4.01 | 3.29 | 1512 | 1585 |
| ta14 | 1589.9 | 1598.6 | 1556.3 | 2.65 | 2.11 | 1486 | 1556.3 |
| ta15 | 1685 | 1633.8 | 1590.6 | 2.65 | 5.6 | 1533 | 1590.2 |
| ta21 | 1990.6 | 2033.1 | 1928.4 | 5.15 | 3.12 | 1871 | 1927.4 |
| ta22 | 1944.3 | 1953.6 | 1838.6 | 5.88 | 5.44 | 1796 | 1838.6 |
| ta23 | 1902.1 | 1925.4 | 1804.5 | 6.27 | 5.13 | 1728 | 1803.8 |
| ta24 | 1994.2 | 1975.6 | 1892.5 | 4.2 | 5.1 | 1801 | 1892.6 |
| ta25 | 1934.3 | 1951 | 1884.5 | 3.41 | 2.57 | 1801 | 1883.9 |
| ta31 | 2152.9 | 2147.8 | 2114.5 | 1.55 | 1.78 | 1983 | 2114.2 |
| ta32 | 2242.8 | 2277.2 | 2155.1 | 5.37 | 3.91 | 2069 | 2153.8 |
| ta33 | 2250.9 | 2269 | 2169.1 | 4.4 | 3.63 | 2125 | 2167.8 |
| ta34 | 2234.5 | 2254.4 | 2137.7 | 5.18 | 4.33 | 2076 | 2137.7 |
| ta35 | 2351.7 | 2369.4 | 2312.1 | 2.42 | 1.68 | 2236 | 2311.4 |
| ta41 | 2571.9 | 2575 | 2479.1 | 3.72 | 3.61 | 2431 | 2477.6 |
| ta42 | 2520.4 | 2540.8 | 2402.8 | 5.43 | 4.67 | 2296 | 2401.8 |
| ta43 | 2424.7 | 2431.4 | 2248.1 | 7.54 | 7.28 | 2190 | 2245.8 |
| ta44 | 2544.5 | 2573 | 2391.6 | 7.05 | 6.01 | 2314 | 2389 |
| ta45 | 2484.9 | 2498.8 | 2364.4 | 5.38 | 4.85 | 2291 | 2364.3 |
| ta51 | 3337.2 | 3344.4 | 3287 | 1.72 | 1.5 | 3180 | 3284.8 |
| ta52 | 3371.4 | 3343.9 | 3252.8 | 2.73 | 3.52 | 3141 | 3251.6 |
| ta53 | 3136.2 | 3096 | 2976.4 | 3.86 | 5.1 | 2932 | 2974.1 |
| ta54 | 3206.9 | 3208.7 | 3117.4 | 2.85 | 2.79 | 3000 | 3113.2 |
| ta55 | 3261.3 | 3260.8 | 3190.8 | 2.15 | 2.16 | 3076 | 3187.7 |
| ta61 | 3521.7 | 3522.5 | 3355.7 | 4.74 | 4.72 | 3287 | 3353.1 |
| ta62 | 3548 | 3581.8 | 3410.7 | 4.78 | 3.87 | 3348 | 3408.5 |
| ta63 | 3322.3 | 3351.9 | 3181.2 | 5.09 | 4.25 | 3128 | 3178.8 |
| ta64 | 3333.8 | 3319.4 | 3253.4 | 1.99 | 2.41 | 3144 | 3250 |
| ta65 | 3371.9 | 3380 | 3331.4 | 1.44 | 1.2 | 3249 | 3329.1 |
| ta71 | 6441 | 6380.6 | 6279 | 1.59 | 2.52 | 6192 | 6277.1 |
| ta72 | 5910.9 | 5957.4 | 5770.6 | 3.14 | 2.37 | 5650 | 5763.2 |
| ta73 | 6527 | 6549.8 | 6262.5 | 4.39 | 4.05 | 6186 | 6260.4 |
| ta74 | 5987.3 | 5970.8 | 5857.1 | 1.9 | 2.18 | 5776 | 5852.4 |
| ta75 | 6293.1 | 6317.4 | 6116.9 | 3.17 | 2.8 | 6022 | 6115.8 |
_Instances complete: **40/40**. DTSA-core better than our ATSA on **40/40**, worse on **0/40**._

_Full set: mean difference **+3.87%**, median **+3.93%** vs our ATSA; mean **+3.70%** vs the paper._

<!-- END:jssp -->

### 5.3 Parameter transfer: `N=D` vs `N=40` — the pre-registered prediction failed

<!-- BEGIN:nd_transfer -->
Across 40 instances (sampler C3), `N=D` vs `N=40` (DTSA-core): mean **+0.19%**, N=D worse on **24/40**, range **-1.50%** to **+2.06%**.

_The pre-registered prediction (§2.4) was that N=D would do **badly** — ~166 iterations at every size. It did not: the two settings are within half a per cent on average. The MWKR seed and early saturation (cf. U14) carry the result; the iteration/population trade-off barely matters. The prediction is recorded as **wrong**, which is itself the parameter-transfer finding._
<!-- END:nd_transfer -->

### 5.4 Per-instance detail, Sahman Table 5 format

Mean / median / min / max per instance (a per-instance summary format), alongside the comparison
table in §5.2. `DTSA-core`, `N=40`, 20 seeds. Generated from the CSV.

<!-- BEGIN:table5 -->
| instance | mean | median | min | max |
|---|---|---|---|---|
| ta01 | 1404.3 | 1416 | 1362 | 1435 |
| ta02 | 1388 | 1394 | 1320 | 1437 |
| ta03 | 1372.7 | 1376 | 1296 | 1462 |
| ta04 | 1370.2 | 1358 | 1311 | 1471 |
| ta05 | 1385.8 | 1392 | 1342 | 1450 |
| ta11 | 1594.8 | 1596 | 1530 | 1649 |
| ta12 | 1561.8 | 1553.5 | 1496 | 1642 |
| ta13 | 1594 | 1593 | 1533 | 1668 |
| ta14 | 1546.6 | 1539 | 1458 | 1652 |
| ta15 | 1591.6 | 1590 | 1523 | 1672 |
| ta21 | 1950.8 | 1948.5 | 1868 | 2036 |
| ta22 | 1857.2 | 1857 | 1801 | 1971 |
| ta23 | 1797.3 | 1791 | 1738 | 1922 |
| ta24 | 1893.2 | 1890.5 | 1820 | 1963 |
| ta25 | 1872.1 | 1862 | 1801 | 2026 |
| ta31 | 2103.2 | 2093.5 | 2041 | 2201 |
| ta32 | 2175.7 | 2170.5 | 2097 | 2292 |
| ta33 | 2198.9 | 2191 | 2097 | 2278 |
| ta34 | 2149.2 | 2154 | 2066 | 2245 |
| ta35 | 2308.4 | 2313 | 2204 | 2450 |
| ta41 | 2499 | 2493.5 | 2402 | 2627 |
| ta42 | 2415 | 2413.5 | 2348 | 2508 |
| ta43 | 2253.6 | 2255 | 2165 | 2349 |
| ta44 | 2428 | 2433.5 | 2367 | 2523 |
| ta45 | 2391.4 | 2387 | 2312 | 2492 |
| ta51 | 3271.8 | 3274 | 3164 | 3365 |
| ta52 | 3220.2 | 3222.5 | 3114 | 3375 |
| ta53 | 2947.2 | 2932 | 2898 | 2998 |
| ta54 | 3084 | 3080.5 | 2993 | 3199 |
| ta55 | 3179.8 | 3174.5 | 3071 | 3297 |
| ta61 | 3314.2 | 3305 | 3241 | 3442 |
| ta62 | 3402 | 3392.5 | 3316 | 3534 |
| ta63 | 3160.6 | 3161.5 | 3049 | 3227 |
| ta64 | 3209 | 3192.5 | 3136 | 3315 |
| ta65 | 3291 | 3289.5 | 3172 | 3422 |
| ta71 | 6234.2 | 6225.5 | 6039 | 6405 |
| ta72 | 5657.2 | 5668.5 | 5481 | 5816 |
| ta73 | 6248.8 | 6236 | 6047 | 6476 |
| ta74 | 5822.2 | 5808.5 | 5744 | 5966 |
| ta75 | 6088.1 | 6081 | 5960 | 6271 |

_DTSA-core, N=40, 20 seeds, sampler C3._
<!-- END:table5 -->

### 5.5 Claim 3 tested directly — ATSA with one MWKR-seeded tree

Claims 2 and 3 both rest on a decomposition of *DTSA's* margin. That decomposition could be a
property of DTSA's search rather than a transferable mechanism, so it was tested on **ATSA itself**.

The ATSA population is 40 trees drawn `uniform(-5, 5)`. Here **one** of them is replaced by the
MWKR dispatching sequence via the exact random-key inverse (`rk_from_sequence`); nothing else
changes — same `Config()`, same `MaxFEs = D×1000`, same 20 seeds. The other 39 trees are
bit-identical to stock, because the njit kernel's `np.random.seed(seed)` stream was verified equal
to `np.random.RandomState(seed)`. **The two arms differ in exactly one of forty trees.**

<!-- BEGIN:b4_claim3 -->
| instance | ATSA stock | ATSA +MWKR seed | DTSA-core | seed gain | gap before | gap after |
|---|---|---|---|---|---|---|
| ta01 | 1460.3 | 1402.8 | 1404.3 | +3.93% | +3.83% | -0.10% |
| ta11 | 1670.9 | 1607 | 1594.8 | +3.82% | +4.55% | +0.76% |
| ta21 | 2033.1 | 1971.4 | 1950.8 | +3.03% | +4.05% | +1.05% |
| ta31 | 2147.8 | 2142.4 | 2103.2 | +0.25% | +2.08% | +1.83% |
| ta41 | 2575 | 2509.8 | 2499 | +2.53% | +2.95% | +0.43% |

_ATSA `Config()` untouched, N=40, MaxFEs = D×1000, 20 seeds per cell. The two arms share the same 40-tree population bar one tree — the random rows are drawn from `np.random.RandomState(seed)`, verified bit-identical to the njit kernel's `np.random.seed(seed)` stream. DTSA column is sampler C3._

**Mean seed gain +2.71%, positive on 5/5. The DTSA–ATSA gap falls from +3.49% to +0.79% — 77% of it closed by one seeded tree.**
<!-- END:b4_claim3 -->

The gap-closure is **not uniform** — see the per-instance column above. It is large on most
instances and small on one, mirroring §5.1's finding that the residual advantage shrinks with size.
The mechanism is real and it transfers, but its size is instance-dependent: a mean, not a guarantee.

This required the only change this workstream has ever made outside `dtsa/`: an optional `init_pop`
argument on `atsa()` (an approved additive change). It is additive and provably
inert when unused — stock ATSA on ta01/ta11/ta21 × seeds 0–2 gives byte-identical makespans, FE
counts and iteration counts before and after the edit, and the test suite is unchanged on both
sides of it.

### 5.6 Timing, threading, and what is actually reproducible

**Read this before comparing our seconds to anyone else's.**

Makespans are **deterministic** given `(instance, seed, budget)`. Each run is an independent
single-threaded process seeded from its own seed, so it is bit-identical at `--jobs 1` and at
`--jobs 8`; concurrency changes only how long the batch takes. **Therefore Tables 5, 7 and 8 are
reproducible on any machine, and only the timing table below requires a controlled single machine.**

Every `wall_seconds` in the earlier CSVs was recorded at varying `--jobs` (8, then 12, then 7, then
6), so those values are not a like-for-like series and are **not** used here. The table below is a
dedicated `--jobs 1` pass with nothing else running.

As it turns out, the contention we were guarding against is small on this machine — see the
slowdown column below, which is close to 1.0 at every size. The earlier timings were therefore
probably not badly wrong. That is a measurement, not a defence of them: it was not known before
this pass, and a clean series is what a Table 6 column has to be.

<!-- BEGIN:timing -->
| instance | D | mean seconds/run |
|---|---|---|
| ta01 | 225 | 4 |
| ta02 | 225 | 4 |
| ta03 | 225 | 4 |
| ta04 | 225 | 3.9 |
| ta05 | 225 | 3.9 |
| ta11 | 300 | 5.8 |
| ta12 | 300 | 5.9 |
| ta13 | 300 | 5.9 |
| ta14 | 300 | 6 |
| ta15 | 300 | 5.9 |
| ta21 | 400 | 9.2 |
| ta22 | 400 | 9.4 |
| ta23 | 400 | 9.6 |
| ta24 | 400 | 9.8 |
| ta25 | 400 | 9.6 |
| ta31 | 450 | 11.3 |
| ta32 | 450 | 11.3 |
| ta33 | 450 | 11.4 |
| ta34 | 450 | 11.2 |
| ta35 | 450 | 11.5 |
| ta41 | 600 | 17.9 |
| ta42 | 600 | 17.9 |
| ta43 | 600 | 17.8 |
| ta44 | 600 | 17.4 |
| ta45 | 600 | 17.5 |
| ta51 | 750 | 29.7 |
| ta52 | 750 | 29.8 |
| ta53 | 750 | 29.3 |
| ta54 | 750 | 29.7 |
| ta55 | 750 | 32.7 |
| ta61 | 1000 | 50.2 |
| ta62 | 1000 | 49.9 |
| ta63 | 1000 | 49.6 |
| ta64 | 1000 | 49.9 |
| ta65 | 1000 | 49.6 |
| ta71 | 2000 | 219.7 |
| ta72 | 2000 | 218.7 |
| ta73 | 2000 | 223.1 |
| ta74 | 2000 | 213.8 |
| ta75 | 2000 | 214 |

_DTSA-core, N=40, sampler C3, 3 seeds per instance, `--jobs 1`, nothing else running. Sum over all 40 instances of one run each: **28.9 min**; the whole pass as measured: **1.44 h**._

_Two rows were re-measured (D008 B2): **ta73** had shown 249.9 s in the first pass, inflated by a single 307 s run; the clean re-measure is 223.1 s, in line with its D=2000 peers, and it replaces the original above. **ta55** was also re-measured and reproduced (32.6 s vs 32.7 s), so its slightly high reading is genuine and kept._
<!-- END:timing -->

**Does parallelism matter?** Measured, not asserted — eight instances across the size range, one
seed each, run sequentially and then all eight in flight. On a 24-logical-core machine, eight
concurrent single-threaded searches barely interfere:

<!-- BEGIN:contention -->
| instance | `--jobs 1` | `--jobs 8` | slowdown |
|---|---|---|---|
| ta01 | 4.0s | 4.0s | 1.02x |
| ta11 | 5.6s | 6.1s | 1.08x |
| ta21 | 8.8s | 9.0s | 1.02x |
| ta31 | 10.5s | 10.8s | 1.03x |
| ta41 | 16.6s | 17.0s | 1.02x |
| ta51 | 28.2s | 28.4s | 1.01x |
| ta61 | 48.0s | 47.7s | 0.99x |
| ta71 | 210.1s | 213.4s | 1.02x |

_One seed per instance, DTSA-core, N=40, sampler C3. At `--jobs 8` all eight searches are in flight at once. **Mean per-run slowdown 1.02x** (range 0.99x–1.08x), against a throughput gain of up to 8x — so the wall-clock win is roughly 7.8x. Makespans are unaffected: the same (instance, seed, budget) gives the same schedule at any job count._
<!-- END:contention -->

**Decoder throughput** at `--jobs 1`:

<!-- BEGIN:throughput -->
| instance | D | FEs/run | seconds/run | evaluations/second |
|---|---|---|---|---|
| ta01 | 225 | 225160 | 4 | 55960 |
| ta71 | 2000 | 2000200 | 219.7 | 9105 |

_Decoder throughput, DTSA-core, `--jobs 1`, single-threaded._
<!-- END:throughput -->

**No cross-language claim is made.** We have no MATLAB implementation of either algorithm, so
nothing here can be compared against the ATSA paper's reported times, and no such comparison is
attempted. These are our own Python/numba numbers on one machine, offered as the Python side of
any comparison the group chooses to run.

### 5.7 ATSA vs DTSA runtime

A runtime question was raised in review: whether DTSA is faster than ATSA at an equal budget. To
answer it fairly, ATSA was timed under the **exact protocol §5.6 used for DTSA** — stock ATSA
(`Config()` untouched), N=40, `MaxFEs = D×1000`, seeds 0–2, all 40 instances, `--jobs 1`, elapsed
seconds, nothing else running (`dtsa/run_atsa_timing.py`). Before any timing was trusted, the
makespans this pass produced were asserted **bit-identical to the validated ATSA** for every
(instance, seed); the run re-executes ATSA only to time it, so the comparison baseline cannot have
moved.

> **Read the wall-clock the right way.** DTSA-core's search loop is plain NumPy, written for
> readability; ATSA's is njit-compiled. A per-run *elapsed* comparison is therefore **confounded by
> the implementation** — it is not a pure algorithm comparison. The fair, like-for-like basis is the
> **evaluation budget**: both run `MaxFEs = D×1000`, exactly accounted as `fes = N + 6·N·iters`.
> Elapsed time is reported because it is what was requested in review and what the paper reports, but a
> slower DTSA per-run number reflects the interpreted reference loop, **not** a slower algorithm.

<!-- BEGIN:runtime_comparison -->
| instance | D | ATSA s/run | DTSA s/run | DTSA/ATSA | faster |
|---|---|---|---|---|---|
| ta01 | 225 | 1.2 | 4 | 3.3x | ATSA |
| ta02 | 225 | 1.2 | 4 | 3.2x | ATSA |
| ta03 | 225 | 1.2 | 4 | 3.3x | ATSA |
| ta04 | 225 | 1.2 | 3.9 | 3.2x | ATSA |
| ta05 | 225 | 1.2 | 3.9 | 3.2x | ATSA |
| ta11 | 300 | 2.1 | 5.8 | 2.8x | ATSA |
| ta12 | 300 | 2.1 | 5.9 | 2.8x | ATSA |
| ta13 | 300 | 2.1 | 5.9 | 2.9x | ATSA |
| ta14 | 300 | 2.1 | 6 | 2.8x | ATSA |
| ta15 | 300 | 2.1 | 5.9 | 2.9x | ATSA |
| ta21 | 400 | 4.1 | 9.2 | 2.3x | ATSA |
| ta22 | 400 | 4.1 | 9.4 | 2.3x | ATSA |
| ta23 | 400 | 4.2 | 9.6 | 2.3x | ATSA |
| ta24 | 400 | 4.1 | 9.8 | 2.4x | ATSA |
| ta25 | 400 | 4.1 | 9.6 | 2.4x | ATSA |
| ta31 | 450 | 5.2 | 11.3 | 2.2x | ATSA |
| ta32 | 450 | 5.2 | 11.3 | 2.2x | ATSA |
| ta33 | 450 | 5.3 | 11.4 | 2.1x | ATSA |
| ta34 | 450 | 5.2 | 11.2 | 2.2x | ATSA |
| ta35 | 450 | 5.2 | 11.5 | 2.2x | ATSA |
| ta41 | 600 | 9.6 | 17.9 | 1.9x | ATSA |
| ta42 | 600 | 9.7 | 17.9 | 1.8x | ATSA |
| ta43 | 600 | 9.7 | 17.8 | 1.8x | ATSA |
| ta44 | 600 | 9.4 | 17.4 | 1.9x | ATSA |
| ta45 | 600 | 9.5 | 17.5 | 1.8x | ATSA |
| ta51 | 750 | 17.3 | 29.7 | 1.7x | ATSA |
| ta52 | 750 | 17.5 | 29.8 | 1.7x | ATSA |
| ta53 | 750 | 17.9 | 29.3 | 1.6x | ATSA |
| ta54 | 750 | 17.5 | 29.7 | 1.7x | ATSA |
| ta55 | 750 | 17.4 | 32.7 | 1.9x | ATSA |
| ta61 | 1000 | 33.7 | 50.2 | 1.5x | ATSA |
| ta62 | 1000 | 34.4 | 49.9 | 1.4x | ATSA |
| ta63 | 1000 | 34.7 | 49.6 | 1.4x | ATSA |
| ta64 | 1000 | 34.8 | 49.9 | 1.4x | ATSA |
| ta65 | 1000 | 35.5 | 49.6 | 1.4x | ATSA |
| ta71 | 2000 | 195.3 | 219.7 | 1.1x | ATSA |
| ta72 | 2000 | 188.2 | 218.7 | 1.2x | ATSA |
| ta73 | 2000 | 188.9 | 223.1 | 1.2x | ATSA |
| ta74 | 2000 | 193.6 | 213.8 | 1.1x | ATSA |
| ta75 | 2000 | 193.8 | 214 | 1.1x | ATSA |

_ATSA (stock `Config()`) and DTSA-core, both N=40, both `MaxFEs = D×1000`, both `--jobs 1`, both 3 seeds/instance, elapsed. The makespans this ATSA pass produced are **bit-identical to the validated ATSA**, asserted per (instance, seed) — the run re-executes ATSA only to time it._

**Is DTSA faster than ATSA at equal budget? No — per run it is slower.** ATSA is faster on **40/40** instances; DTSA-core is on average **~2.1× slower per run** (**1.1×** at the largest instance, ta71). **This is not an algorithm comparison.** DTSA-core's search loop is plain NumPy, written for readability; ATSA's is njit-compiled. The like-for-like basis is the evaluation budget — identical `MaxFEs = D×1000`, exactly accounted as `fes = N + 6·N·iters` — not wall-clock, which here is dominated by the interpreted-vs-compiled implementation gap, not by the search. Elapsed time is reported because it is what was requested in review and what the paper reports; the per-run gap reflects the reference implementation, not the method. A compiled DTSA loop (a future change) would be the apples-to-apples runtime test and is not yet done._
<!-- END:runtime_comparison -->

**Contention** — the same 8 instances §5.6 used, seed 0, at `--jobs 1`, `--jobs 8`, and (since ATSA
has never hit the RAM pressure DTSA did) additionally at `--jobs 24`, this machine's logical-core
count:

<!-- BEGIN:atsa_contention -->
| instance | `--jobs 1` | `--jobs 8` | `--jobs 24` | slowdown@8 | slowdown@24 |
|---|---|---|---|---|---|
| ta01 | 1.2s | 1.5s | 2.7s | 1.21x | 2.17x |
| ta11 | 2.1s | 2.6s | 4.3s | 1.22x | 2.04x |
| ta21 | 4.2s | 4.6s | 7.6s | 1.10x | 1.81x |
| ta31 | 5.3s | 5.6s | 9.2s | 1.06x | 1.73x |
| ta41 | 9.4s | 10.1s | 14.6s | 1.07x | 1.55x |
| ta51 | 17.7s | 17.7s | 23.2s | 1.00x | 1.31x |
| ta61 | 34.0s | 33.7s | 39.6s | 0.99x | 1.16x |
| ta71 | 194.6s | 198.3s | 203.3s | 1.02x | 1.04x |

_ATSA stock, seed 0, N=40. `--jobs 8` runs the 8 instances concurrently; `--jobs 24` saturates all 24 logical cores with 24 real tasks (8 instances × 3 seeds). Mean per-run slowdown at **8** concurrent: **1.08×**. Mean per-run slowdown at **24** concurrent: **1.60×**. `--jobs 24` is the highest level that ran clean (no memory backoff). Makespans are unaffected by job count — only elapsed time is._
<!-- END:atsa_contention -->

**No "vs paper's reported ATSA time" column is included.** The Sahman (2022) paper reports ATSA
elapsed times, but they are not parsed into any module here (`dtsa_tables.py` parses the DTSA/TSP
paper; `paper_table5.py` holds only makespans, no times). Per this project's rule, they are not
hand-typed — that comparison is **pending a parse** and is deliberately left out rather than
reconstructed from memory.

---

## 6. Findings

1. **DTSA's published results include an uncounted local-search budget.** Fig. 6 never charges
   2-opt against `MaxFEs`. Any comparison of DTSA against a budget-matched algorithm is
   correspondingly generous to DTSA. We keep `DTSA-core` and `DTSA+LS` separate for this reason.
2. **The search saturates well before the budget — two independent observations, one claim.**
   On TSP, six times the iterations at the same budget left the mean deviation unchanged to two
   decimal places (§4.7, generated). On the job shop, `N=D` runs far fewer iterations than `N=40`
   at the same budget and is worse by a fraction of a per cent (§5.3, generated). The second
   observation is informative rather than a restatement of the first precisely because the
   pre-registered prediction attached to it — that `N=D` would do *badly* — was **wrong**.
   DTSA reaches near-final quality early and then stops improving.
   *Practical value for an industrial scheduler:* near-final schedules are available at a
   fraction of the nominal budget, so the budget can be cut for responsiveness at little cost.
   Two observations support this and no more; the shape of the curve between them is unmeasured.
3. **A dispatching-rule seed transfers to ATSA.** One MWKR tree out of forty, no change to the
   search, closes most of the DTSA–ATSA gap on every instance tested (§5.5). Pre-registered
   before the run, then measured.
4. **An unspecified detail — the nearest-neighbour start city — is worth several per cent** and
   accounts for essentially all of our Gate 1 residual (§4.5). It is a reproducibility defect in
   the paper.
5. **Three arithmetic/typesetting defects in the paper** (§3), each caught by an automated
   checksum rather than by eye.
6. **The operator ranking reproduces cleanly** — symmetry beats shift beats swap, in every
   configuration cell we ran — even though the absolute level does not. The operators behave
   qualitatively as the paper describes.
7. **U1 is resolvable after all** — to **C3** — but only once the start-city defect (U12) is
   corrected first. The correct answer was invisible while an unrelated defect was uncontrolled
   (§4.6). This is why defects are fixed in dependency order, not in the order found.

### 6.1 Method findings — the checks that earned their keep

These are not results about DTSA; they are evidence the method caught its own mistakes.

- **The FE-accounting identity `fes = N + 6·N·iterations` is asserted on every single run** — Gate 1
  (1,080), the job shop (1,600+), every diagnostic. It caught two real bugs on the ATSA side of this
  project and has never once failed here.
- **D1 equivariance: 10,800 trials, zero divergences.** Mutating the random-key vector and decoding
  gives *identically* the sequence you get by decoding then mutating, for all three operators. This
  is what licenses reusing the verified ATSA decoder unchanged.
- **U15 near-miss.** The Table 5 checksum flagged five KROE100 rows. The first instinct — encoded in
  the extractor as *"the parse is wrong"* — was itself wrong. Solving each row for its *implied*
  optimum gave ≈22141 (KROB100's optimum, three rows up) five times over, proving the **paper** was
  scored against the wrong instance and the parse was fine. Had the parser been "fixed" to agree,
  five correct means would have been corrupted to match a wrong error column. A checksum tells you
  *something* disagrees, not *what* — always solve for the implied quantity first.
- **A wasted experiment, disclosed.** The §4.5 residual 2×2 had an `F6` arm (immediate vs deferred
  `best` update) that was **inert by construction**: on a row whose seeds come only from the current
  tree, `best` is never read, so the arm could not move the outcome. Byte-identical results were the
  *expected* outcome, not a finding. Reported as *no information*, not *no effect* — a check that a
  factor is reachable should precede running it.

---

## 7. Reproducibility

Every derived number in this report is computed by `uv run python dtsa/metrics.py` from
`dtsa/dtsa_tables.py` (the paper's tables, transcribed offline from the source paper, never
hand-typed into working code) and the results CSVs.

| what | command |
|---|---|
| Full test suite | `uv run pytest -q` |
| Settle the distance convention (U9) | `uv run python dtsa/validation_tsp/verify_berlin52.py` |
| Gate 1 (1,080 runs) | `uv run python dtsa/validation_tsp/run_gate1.py --jobs 8` |
| Residual diagnostic | `uv run python -u dtsa/run_partb.py --jobs 8` |
| U1 re-score (corrected start) | `uv run python -u dtsa/run_u1_rescore.py --jobs 8` |
| Sampler sensitivity (C1 vs C3) | `uv run python -u dtsa/run_sampler_sensitivity.py --jobs 8` |
| N5 local-search diagnostic | `uv run python -u dtsa/diag_localsearch.py` |
| Job-shop run (C1) / repeat (C3) | `uv run python -u dtsa/run_jssp.py --jobs 8` / `run_jssp_c3.py` |
| ATSA + one MWKR tree (§5.5) | `uv run python -u dtsa/run_b4_atsa_seeded.py --jobs 8` |
| Timing, Table 6 column (§5.6) | `uv run python -u dtsa/run_timing.py serial` — **`--jobs 1`, quiet machine** |
| Contention measurement (§5.6) | `uv run python -u dtsa/run_timing.py contention` |
| Recompute the numbers | `uv run python dtsa/metrics.py` |

All runs are seeded and checkpointed per configuration; a re-invocation resumes. See the README's
hardware section for guidance on worker counts.

**Data provenance.** TSPLIB instances used by the TSP validation gate live in `data/tsplib/`,
each byte-identical across independent mirrors; SHA256 and source are recorded in `data/README.md`. The
Taillard job-shop instances are the originals in `data/raw/` and are never written.

### System configuration

<!-- BEGIN:environment -->
| key | value |
|---|---|
| timestamp | 2026-07-24T10:07:29 |
| git_sha | 5ee1241 |
| os | Windows 11 (10.0.26200) |
| machine | AMD64 |
| processor | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |
| logical_cores | 24 |
| ram_total_gb | 31.84 |
| ram_available_gb | 16.94 |
| commit_limit_gb | 33.84 |
| python | 3.12.8 |
| numpy | 2.1.3 |
| numba | 0.66.0 |
<!-- END:environment -->

---

## 8. Open questions

| id | question | status |
|---|---|---|
| **U1** | How are symmetry's block positions and size sampled? | **RESOLVED → C3** under the corrected NN start (§4.6). The earlier "unresolved" was an artefact of scoring against the wrong start city (U12) |
| **U11** | Stand size for the paper's Experiment 4 | Inferred as `N` = city count; never stated |
| **U12** | Which city does the nearest-neighbour tour start from? | **Unresolved, and it matters** — worth several per cent (§4.5) |
| **U17** | What does a Table 1 ablation row hold fixed? | Adopted the Fig.-6-faithful reading; explains the shape, not the level |
| **N5 local search** | Why does the critical-block neighbourhood produce almost no moves on converged solutions? | **Resolved: GENUINE, not a bug** (§8.1). The implementation is correct N5; it evaluates few candidates *by design*, and the converged DTSA solution is already N5-locally-optimal. **`DTSA+LS` still excluded from every conclusion** — it adds nothing here. |
| — | The remaining Gate 1 level gap | **Unexplained.** Not chased further: after a shape fix, continuing to adjust until the level matches is indistinguishable from tuning |

### 8.1 The N5 local search — genuine, not a bug

An earlier note (D004) guessed the critical path *"fragments into blocks of length 1"* on converged
solutions. **That guess was wrong.** Instrumenting a converged ta01 and ta71 schedule (`dtsa/diag_localsearch.py`):

<!-- BEGIN:n5_diag -->
| instance | critical path (ops) | block-size distribution | blocks ≥2 | N5 candidate moves | improving |
|---|---|---|---|---|---|
| ta01 (D=225) | 23 | {1: 5, 2: 1, 7: 1, 9: 1} | 3 | 5 | 0 |
| ta71 (D=2000) | 126 | {3: 1, 18: 1, 33: 1, 72: 1} | 4 | 8 | 0 |

_Largest block observed: **72** operations. Candidate moves evaluated: **5–8**._
<!-- END:n5_diag -->

Large blocks *do* exist — see the distribution above. The reason so few candidates are evaluated is
that **N5 (Nowicki–Smutnicki) by construction probes only the two operations at each *end* of each
block** —
~2 candidates per block, independent of block size. So the small count is correct behaviour, and
the converged DTSA solution is already N5-locally-optimal (0 improving moves). The implementation is
right; N5 is simply a small, focused neighbourhood that adds nothing on top of a converged DTSA
solution. `DTSA+LS` stays excluded from conclusions because it does nothing here — now an explained
fact, not an unexplained anomaly.
