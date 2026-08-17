#!/usr/bin/env python3
"""
Judge Gate 1 against the tiering pre-registered in the DTSA adaptation notes D5 at commit d63900a,
and score U1 on the two criteria added at 06f6700.

The tiering is NOT recomputed or reinterpreted here. HARD is judged at the LITERAL DEFAULTS --
NS = 6 (Fig. 6 line 12) and sampler C1 (the sampler design notes's maximally literal reading) -- because
"defaults stay literal" is the standing rule. Every other cell is REPORTED.

Usage:  uv run python dtsa/analyse_gate1.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dtsa_tables import TABLE1                                  # noqa: E402

CSV = pathlib.Path(__file__).resolve().parent.parent.parent / "results" / "gate1" / "gate1.csv"
TOL = 1.5                                    # per cent; OUR construction, not the paper's (§5.6)
LITERAL_NS, LITERAL_SAMPLER = 6, "C1"
EXCLUDED = "swap(best tree)"                 # U16 -- duplicated row, no independent value


def row_key(op: str, src: str) -> str:
    return f"{op}({src} tree)"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["row"] = df.apply(lambda r: row_key(r["operator"], r["source"]), axis=1)
    df["published_pre"] = df["row"].map(lambda k: TABLE1[k]["tsa_mean"])
    df["published_post"] = df["row"].map(lambda k: TABLE1[k]["tsa_2opt_mean"])
    return df


def agg(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["row", "operator", "source", "NS", "sampler"], as_index=False).agg(
        ours_pre=("pre_2opt", "mean"),
        ours_post=("post_2opt", "mean"),
        pre_std=("pre_2opt", "std"),
        post_std=("post_2opt", "std"),
        runs_2opt_improved=("two_opt_moves", lambda s: int((s > 0).sum())),
        two_opt_evals=("two_opt_evals", "mean"),
        published_pre=("published_pre", "first"),
        published_post=("published_post", "first"),
        n=("seed", "count"),
    )
    g["dev_pct"] = (g["ours_pre"] - g["published_pre"]) / g["published_pre"] * 100
    g["our_2opt_gain_pct"] = (g["ours_pre"] - g["ours_post"]) / g["ours_pre"] * 100
    g["paper_2opt_gain_pct"] = (g["published_pre"] - g["published_post"]) / g["published_pre"] * 100
    return g


def main() -> None:
    df = load()
    g = agg(df)
    assert (g["n"] == 30).all(), "every group must have 30 runs"
    print(f"Gate 1: {len(df)} runs, {len(g)} configuration groups, 30 seeds each\n")

    # ==========================================================================================
    print("=" * 94)
    print("1. HARD CRITERIA -- judged at the LITERAL defaults: NS = 6, sampler C1")
    print("=" * 94)
    lit = g[(g["NS"] == LITERAL_NS) &
            ((g["sampler"] == LITERAL_SAMPLER) | (g["sampler"] == "-"))]

    best_per_op = lit.groupby("operator")["ours_pre"].min().sort_values()
    ranking = list(best_per_op.index)
    want = ["symmetry", "shift", "swap"]
    i_pass = ranking == want
    print(f"\n  (i)  operator ranking on tsa_mean (best row per operator)")
    for op in ranking:
        print(f"         {op:<9} {best_per_op[op]:9.2f}")
    print(f"       ours   : {' < '.join(ranking)}")
    print(f"       want   : {' < '.join(want)}   (§5.1's own claim)")
    print(f"       -> {'PASS' if i_pass else 'FAIL'}")

    anchor = lit[lit["row"] == "symmetry(current tree)"].iloc[0]
    ii_pass = abs(anchor["dev_pct"]) <= TOL
    print(f"\n  (ii) symmetry(current tree) tsa_mean within +/-{TOL}%")
    print(f"       ours {anchor['ours_pre']:.2f}   published {anchor['published_pre']:.2f}   "
          f"deviation {anchor['dev_pct']:+.2f}%")
    print(f"       -> {'PASS' if ii_pass else 'FAIL'}")

    print(f"\n  GATE 1 (literal defaults): {'PASS' if (i_pass and ii_pass) else 'FAIL'}")

    # same two criteria at every other (NS, sampler) cell -- reported, never gated
    print("\n  The same two criteria at the other cells (REPORTED, not gated):")
    print(f"    {'NS':>3} {'sampler':<8} {'(i) ranking':<28} {'(ii) sym(current) dev':>22}")
    for ns in sorted(g["NS"].unique(), reverse=True):
        for sampler in ["C1", "C2", "C3", "C4"]:
            cell = g[(g["NS"] == ns) &
                     ((g["sampler"] == sampler) | (g["sampler"] == "-"))]
            if cell.empty:
                continue
            bp = cell.groupby("operator")["ours_pre"].min().sort_values()
            rk = " < ".join(bp.index)
            a = cell[cell["row"] == "symmetry(current tree)"]
            dev = a.iloc[0]["dev_pct"] if len(a) else float("nan")
            mark_i = "OK " if list(bp.index) == want else "no "
            mark_ii = "OK " if abs(dev) <= TOL else "no "
            print(f"    {ns:>3} {sampler:<8} {mark_i}{rk:<25} {mark_ii}{dev:+8.2f}%")

    # ==========================================================================================
    print("\n" + "=" * 94)
    print("2. REPORTED -- all rows, both NS settings, at the literal sampler C1")
    print("=" * 94)
    print(f"\n  {'row':<24} {'NS':>3} {'ours pre':>10} {'published':>10} {'dev%':>8} "
          f"{'ours post':>10} {'pub post':>9}  note")
    for ns in [6, 1]:
        for row in sorted(TABLE1):
            cell = g[(g["row"] == row) & (g["NS"] == ns) &
                     ((g["sampler"] == LITERAL_SAMPLER) | (g["sampler"] == "-"))]
            if cell.empty:
                continue
            r = cell.iloc[0]
            note = "EXCLUDED (U16)" if row == EXCLUDED else ""
            print(f"  {row:<24} {ns:>3} {r['ours_pre']:>10.2f} {r['published_pre']:>10.2f} "
                  f"{r['dev_pct']:>+8.2f} {r['ours_post']:>10.2f} {r['published_post']:>9.2f}"
                  f"  {note}")

    usable = g[(g["row"] != EXCLUDED) & (g["NS"] == LITERAL_NS) &
               ((g["sampler"] == LITERAL_SAMPLER) | (g["sampler"] == "-"))]
    print(f"\n  8 usable rows at NS=6/C1: mean |dev| {usable['dev_pct'].abs().mean():.2f}%, "
          f"worst {usable['dev_pct'].abs().max():.2f}%, "
          f"{(usable['dev_pct'].abs() <= TOL).sum()}/8 within +/-{TOL}%")

    # ==========================================================================================
    print("\n" + "=" * 94)
    print("3. U1 -- all four candidates on BOTH criteria (a) and (b). Rejects included.")
    print("=" * 94)
    paper_gain = g["paper_2opt_gain_pct"][g["row"] == "symmetry(current tree)"].iloc[0]
    print(f"\n  criterion (b) target: paper's symmetry(current tree) 2-opt gain = "
          f"{paper_gain:.2f}%\n")
    for ns in [6, 1]:
        print(f"  NS = {ns}")
        print(f"    {'cand':<5} {'(a) MAD over 3 sym rows':>24} {'sym(cur) dev%':>14} "
              f"{'(b) our 2opt gain%':>19} {'runs 2opt moved':>17}")
        for sampler in ["C1", "C2", "C3", "C4"]:
            sym = g[(g["operator"] == "symmetry") & (g["sampler"] == sampler) & (g["NS"] == ns)]
            if sym.empty:
                continue
            mad = (sym["ours_pre"] - sym["published_pre"]).abs().mean()
            dev = sym[sym["row"] == "symmetry(current tree)"].iloc[0]["dev_pct"]
            gain = sym[sym["row"] == "symmetry(current tree)"].iloc[0]["our_2opt_gain_pct"]
            moved = sym[sym["row"] == "symmetry(current tree)"].iloc[0]["runs_2opt_improved"]
            print(f"    {sampler:<5} {mad:>24.2f} {dev:>+13.2f}% {gain:>18.2f}% "
                  f"{moved:>14}/30")
        print()

    # ==========================================================================================
    print("=" * 94)
    print("4. U14 -- which NS reproduces Table 1? (prediction: NS=1, pre-registered at 06f6700)")
    print("=" * 94)
    for ns in [6, 1]:
        cell = g[(g["NS"] == ns) & (g["row"] != EXCLUDED) &
                 ((g["sampler"] == LITERAL_SAMPLER) | (g["sampler"] == "-"))]
        print(f"\n  NS = {ns}: mean |dev| over the 8 usable rows = "
              f"{cell['dev_pct'].abs().mean():6.2f}%   "
              f"({(cell['dev_pct'].abs() <= TOL).sum()}/8 within +/-{TOL}%)   "
              f"iterations = {int(df[df['NS'] == ns]['iterations'].iloc[0])}")

    # ==========================================================================================
    print("\n" + "=" * 94)
    print("5. U16 free check -- do OUR swap(current) and swap(best) collide?")
    print("=" * 94)
    for ns in [6, 1]:
        a = g[(g["row"] == "swap(current tree)") & (g["NS"] == ns)].iloc[0]
        b = g[(g["row"] == "swap(best tree)") & (g["NS"] == ns)].iloc[0]
        print(f"\n  NS = {ns}")
        print(f"    swap(current tree): pre {a['ours_pre']:9.2f}  post {a['ours_post']:9.2f}  "
              f"post-std {a['post_std']:7.2f}")
        print(f"    swap(best tree)   : pre {b['ours_pre']:9.2f}  post {b['ours_post']:9.2f}  "
              f"post-std {b['post_std']:7.2f}")
        print(f"    identical? {'YES' if abs(a['ours_pre'] - b['ours_pre']) < 0.005 else 'NO'}"
              f"   (paper's two rows are identical to 2 dp, std 0.00)")

    g.to_csv(CSV.parent / "gate1_summary.csv", index=False)
    print(f"\nwrote {CSV.parent / 'gate1_summary.csv'}")


if __name__ == "__main__":
    main()
