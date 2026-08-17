"""
Quick ours-vs-paper summary, printed to the console.

NOTE ON PROVENANCE. This file once carried a hand-transcribed copy of the paper's Table 5 that was
wrong on 30 of 40 rows (every instance from ta21 onward). Those errors grew with problem size and
manufactured an entirely false "our reproduction degrades with problem size" finding. The truth is
no drift (corr(D, diff%) = -0.02).

The paper values are now IMPORTED from src/atsa_jssp/paper_table5.py (a committed transcription,
checksummed against the paper's own Table 5 AVG row by tests/test_paper_table5.py). Never
re-introduce a hand-typed paper dict here or anywhere else.
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# our runs: ta01 + the ta02..ta80 campaign
df = pd.concat([
    pd.read_csv(ROOT / "results/atsa/atsa_ta01.csv"),
    pd.read_csv(ROOT / "results/atsa/atsa_ta02_ta80.csv"),
], ignore_index=True)

from atsa_jssp.paper_table5 import TABLE5  # noqa: E402

paper = {i: (v["ATSA"]["mean"], v["ATSA"]["min"]) for i, v in TABLE5.items()}

g = df.groupby('instance')['cmax']
summary = pd.DataFrame({
    'mean': g.mean().round(1),
    'min':  g.min(),
    'max':  g.max(),
    'std':  g.std().round(1),
    'wall': df.groupby('instance')['wall_s'].mean().round(1),
})

print("\n=== PAPER INSTANCES — OUR vs PAPER ===")
print(f"{'inst':<6} {'OurMean':>9} {'PaperMean':>10} {'Diff%':>7} {'OurMin':>7} {'PaperMin':>9} {'OurMax':>7} {'Std':>6}")
print("-" * 65)

diffs = []
for inst in sorted(paper.keys()):
    if inst not in summary.index:
        continue
    r = summary.loc[inst]
    pm, pmin = paper[inst]
    diff = (r['mean'] - pm) / pm * 100
    diffs.append(diff)
    flag = "OK" if abs(diff) < 2 else ("hi" if diff > 0 else "lo")
    print(f"{inst:<6} {r['mean']:>9.1f} {pm:>10.1f} {diff:>+7.2f}% {r['min']:>7} {pmin:>9} {r['max']:>7} {r['std']:>6}  {flag}")

print(f"\nAvg |diff| from paper mean: {sum(abs(d) for d in diffs)/len(diffs):.2f}%")
print(f"Max |diff|: {max(abs(d) for d in diffs):.2f}%")
print(f"Instances within +/-2% of paper: {sum(1 for d in diffs if abs(d)<2)}/{len(diffs)}")
print(f"Instances within +/-5% of paper: {sum(1 for d in diffs if abs(d)<5)}/{len(diffs)}")

print("\n=== NON-PAPER INSTANCES (extra coverage; not a paper comparison) ===")
non_paper = [i for i in summary.index if i not in paper]
print(f"{'inst':<6} {'Mean':>8} {'Min':>6} {'Max':>6} {'Std':>6} {'s/run':>6}")
print("-" * 40)
for inst in sorted(non_paper):
    r = summary.loc[inst]
    print(f"{inst:<6} {r['mean']:>8.1f} {r['min']:>6} {r['max']:>6} {r['std']:>6} {r['wall']:>6.1f}")

print(f"\n=== OVERALL SUMMARY ===")
print(f"Total instances: {len(summary)}/80")
print(f"Total runs: {len(df)}")
print(f"All FE budgets respected: {(df['fes_used'] <= df['max_fes'] + 800).all()}")
