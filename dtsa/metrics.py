#!/usr/bin/env python3
"""
THE single source of every derived number in the DTSA report and the DTSA PDF.

WHY THIS EXISTS (D008 Part A): the report generators (not included in this release) each carried their
own prose and each computed its own figures. They drifted -- the PDF quoted the DTSA advantage
as 3.77% in one section and 3.49% one page later, the same *kind* of quantity over two different
instance sets and two different samplers, with neither labelled. Two computations of one number
is the structural fault; a wrong number is only the symptom.

So: every scalar is computed HERE, exactly once, and both generators import it. Neither generator
may contain a numeric literal in prose.

SCOPE LABELS ARE PART OF THE VALUE. Three different "DTSA advantage" figures are legitimately in
circulation and they are not interchangeable:

    all40_c3      +4.06%  -- 40 instances, sampler C3, DTSA-core vs our ATSA.  THE headline.
    b4_5_c3       +3.49%  -- the 5 instances of the B4 experiment, sampler C3, vs stock ATSA.
    mwkr5_c1      +3.77%  -- the 5 instances of the D005 MWKR ablation, sampler C1. HISTORICAL:
                             the unseeded arm only exists at C1, so the ablation stays a C1
                             experiment and is reported as one. It is not stale, it is a
                             different scope, and it must never be quoted without saying so.

Every scope-bearing entry therefore carries a `.scope` string, and the generators are expected to
print it. Usage:  from metrics import M; M()["jssp"]["N=40"].adv_ours
"""
from __future__ import annotations

import dataclasses
import functools
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"                                  # split into atsa/ dtsa/ ablations/ timing/ gate1/
sys.path.insert(0, str(HERE))
from dtsa_tables import OPTIMA, TABLE1                       # noqa: E402

TOL = 1.5                     # our Gate 1 tolerance, not the paper's (report 4.2)
MWKR_THRESHOLD = 50.0         # pre-registered in D005 before the ablation
SEEDS_PER_CELL = 20


@dataclasses.dataclass(frozen=True)
class Adv:
    """A DTSA-advantage figure that knows its own scope."""
    value: float
    n_instances: int
    sampler: str
    what: str

    @property
    def scope(self) -> str:
        return f"{self.n_instances} instances, sampler {self.sampler}"

    def __str__(self) -> str:
        return f"{self.value:+.2f}% ({self.scope})"


def _read(name: str) -> pd.DataFrame | None:
    # results/ is split into subfolders (atsa/ dtsa/ ablations/ timing/ gate1/); filenames are
    # unique across them, so resolve by searching the whole tree.
    hits = list(RESULTS.glob(f"**/{name}"))
    return pd.read_csv(hits[0]) if hits else None


def _our_atsa() -> pd.Series | None:
    """Our own ATSA 20-run means, from the canonical pair of CSVs (ta01 + the combined set)."""
    parts = [pd.read_csv(REPO / rel) for rel in
             ("results/atsa/atsa_ta01.csv", "results/atsa/atsa_ta02_ta80.csv")
             if (REPO / rel).exists()]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).groupby("instance")["cmax"].mean()


def _paper_atsa() -> dict:
    sys.path.insert(0, str(HERE.parent / "src"))
    from atsa_jssp.paper_table5 import TABLE5 as T5
    return {k: v["ATSA"]["mean"] for k, v in T5.items()}


def _jssp_source() -> tuple[pd.DataFrame | None, str]:
    """U1 resolved to C3 (D006): prefer the completed C3 run, fall back to C1."""
    c3 = _read("dtsa_jssp_c3.csv")
    if c3 is not None and len(c3):
        return c3, "C3"
    c1 = _read("dtsa_jssp.csv")
    return (c1, "C1") if c1 is not None else (None, "-")


@functools.lru_cache(maxsize=1)
def M() -> dict:
    m: dict = {}
    df, sampler = _jssp_source()
    ours, paper = _our_atsa(), _paper_atsa()
    m["sampler"] = sampler

    # ---- job shop, all 40 instances -------------------------------------------------
    m["jssp"] = {}
    if df is not None and ours is not None:
        core = df[df.config == "DTSA-core"]
        for ns in ("N=40", "N=D"):
            sub = core[core.n_setting == ns].groupby("instance")["cmax"].mean()
            idx = [i for i in sub.index if i in paper and i in ours.index]
            if not idx:
                continue
            m["jssp"][ns] = dict(
                adv_ours=Adv(sum((ours[i] - sub[i]) / ours[i] * 100 for i in idx) / len(idx),
                             len(idx), sampler, "DTSA-core vs our ATSA"),
                adv_paper=Adv(sum((paper[i] - sub[i]) / paper[i] * 100 for i in idx) / len(idx),
                              len(idx), sampler, "DTSA-core vs published ATSA"),
                wins=sum(1 for i in idx if sub[i] < ours[i]),
                n=len(idx), means=sub)
        g = core.groupby(["instance", "n_setting"]).size()
        m["complete_groups"] = int((g >= SEEDS_PER_CELL).sum())
        m["n_instances"] = core.instance.nunique()
    m["seeds_per_cell"] = SEEDS_PER_CELL

    # ---- our-ATSA-vs-Sahman reproduction fidelity (the credibility floor, B2) --------
    # The whole DTSA-vs-ATSA comparison rests on "our ATSA" being a faithful reproduction
    # of Sahman (2022) Table 5. Quantify it once, from the same read-only CSVs.
    if ours is not None and paper:
        dev = [(ours[i] - paper[i]) / paper[i] * 100 for i in paper if i in ours.index]
        ab = [abs(x) for x in dev]
        m["atsa_repro"] = dict(n=len(dev), max_abs=max(ab), mad=sum(ab) / len(ab),
                               signed_mean=sum(dev) / len(dev),
                               within=[x for x in (3.0, 3.1, 3.5, 4.0) if all(a <= x for a in ab)])

    # ---- N=D vs N=40 (the pre-registered prediction that failed) ---------------------
    if {"N=40", "N=D"} <= set(m.get("jssp", {})):
        a, b = m["jssp"]["N=40"]["means"], m["jssp"]["N=D"]["means"]
        common = [i for i in a.index if i in b.index]
        d = [(b[i] - a[i]) / a[i] * 100 for i in common]
        m["nd"] = dict(mean=sum(d) / len(d), worse=sum(1 for v in d if v > 0),
                       n=len(d), lo=min(d), hi=max(d))

    # ---- the D005 MWKR ablation: 5 instances, sampler C1, HISTORICAL scope ----------
    ab = _read("mwkr_ablation.csv")
    if ab is not None and ours is not None:
        ins = sorted(ab.instance.unique())
        smp = str(ab["sampler"].iloc[0]) if "sampler" in ab else "C1"
        # Round each per-instance % to 2 dp BEFORE averaging, matching report §5.1
        # (the report generator): the report's table shows the rounded per-instance values
        # and its mean is the mean of those, so the slide must use the same definition.
        se = [round((ours[i] - ab[(ab.instance == i) & ab.seeded_with_mwkr].cmax.mean())
                    / ours[i] * 100, 2) for i in ins]
        un = [round((ours[i] - ab[(ab.instance == i) & ~ab.seeded_with_mwkr].cmax.mean())
                    / ours[i] * 100, 2) for i in ins]
        gs, gu = sum(se) / len(se), sum(un) / len(un)
        m["mwkr"] = dict(
            instances=ins, seeded=se, unseeded=un,
            gap_seeded=Adv(gs, len(ins), smp, "DTSA seeded vs our ATSA"),
            gap_unseeded=Adv(gu, len(ins), smp, "DTSA unseeded vs our ATSA"),
            retained=gu / gs * 100, threshold=MWKR_THRESHOLD,
            wins_unseeded=sum(1 for v in un if v > 0),
            first=un[0], last=un[-1], first_i=ins[0], last_i=ins[-1], sampler=smp)

    # ---- B4: ATSA + one MWKR tree, 5 instances, DTSA column at C3 -------------------
    b4 = _read("b4_atsa_seeded.csv")
    if b4 is not None and "jssp" in m and "N=40" in m["jssp"]:
        piv = b4.pivot_table(index="instance", columns="arm", values="cmax", aggfunc="mean")
        dt = m["jssp"]["N=40"]["means"]
        rows, gains, gs_, gm_ = [], [], [], []
        for i in sorted(piv.index):
            if i not in dt.index:
                continue
            st, mw, d0 = piv.loc[i, "stock"], piv.loc[i, "mwkr1"], dt[i]
            gains.append((st - mw) / st * 100)
            gs_.append((st - d0) / st * 100)
            gm_.append((mw - d0) / mw * 100)
            rows.append(dict(instance=i, stock=st, mwkr1=mw, dtsa=d0, gain=gains[-1],
                             before=gs_[-1], after=gm_[-1]))
        n = len(rows)
        before = sum(gs_) / n
        after = sum(gm_) / n
        m["b4"] = dict(
            rows=rows, n=n, seeds=SEEDS_PER_CELL,
            gain=sum(gains) / n, positive=sum(1 for g in gains if g > 0),
            gap_before=Adv(before, n, sampler, "DTSA vs stock ATSA"),
            gap_after=Adv(after, n, sampler, "DTSA vs seeded ATSA"),
            closed=(1 - after / before) * 100 if before else float("nan"))

    # ---- Gate 1 --------------------------------------------------------------------
    g1 = _read("gate1.csv")
    if g1 is not None:
        g1["row"] = g1["operator"] + "(" + g1["source"] + " tree)"
        lit = g1[(g1.NS == 6) & (g1.sampler.isin(["C1", "-"]))]
        by = lit.groupby(["row", "operator"])["pre_2opt"].mean().reset_index()
        best = by.groupby("operator")["pre_2opt"].min().sort_values()
        anchor = by[by["row"] == "symmetry(current tree)"]["pre_2opt"].iloc[0]
        pub = TABLE1["symmetry(current tree)"]["tsa_mean"]
        m["gate1"] = dict(anchor=anchor, published=pub, dev=(anchor - pub) / pub * 100,
                          order=list(best.index), best=best, tol=TOL, seeds=g1.seed.nunique(),
                          n_runs=len(g1), configs=lit["row"].nunique())
        ns1 = g1[(g1.NS == 1) & (g1.sampler.isin(["C1", "-"]))]
        if len(ns1):
            def _dev(sub):
                gg = sub.groupby("row")["pre_2opt"].mean()
                gg = gg[gg.index != "swap(best tree)"]
                return sum(abs((v - TABLE1[k]["tsa_mean"]) / TABLE1[k]["tsa_mean"] * 100)
                           for k, v in gg.items()) / len(gg)
            m["u14"] = dict(dev6=_dev(lit), dev1=_dev(ns1),
                            iters6=int(lit.iterations.iloc[0]),
                            iters1=int(ns1.iterations.iloc[0]))

    # ---- U12: what the unspecified start city is worth ------------------------------
    pb = _read("partb_residual.csv")
    if pb is not None:
        g = pb.groupby("u12_nn").agg(pre=("pre_2opt", "mean"), nn=("nn_seed_length", "first"))
        if {"city_0", "best_of_52"} <= set(g.index):
            pub = TABLE1["symmetry(current tree)"]["tsa_mean"]
            n0, nb = float(g.loc["city_0", "nn"]), float(g.loc["best_of_52", "nn"])
            d0 = (g.loc["city_0", "pre"] - pub) / pub * 100
            db = (g.loc["best_of_52", "pre"] - pub) / pub * 100
            m["u12"] = dict(
                nn_city0=n0, nn_best=nb, dev_city0=d0, dev_best=db,
                # DENOMINATOR IS EXPLICIT (D008 B3): shorter *than the city-0 tour*.
                shorter_pct_of_city0=(n0 - nb) / n0 * 100,
                pp_move=abs(d0 - db), n_starts=int(OPTIMA["BERLIN52"] and 52))

    # ---- timing ---------------------------------------------------------------------
    ts = _read("timing_serial.csv")
    if ts is not None:
        t = ts[ts.jobs == 1]
        g = t.groupby("instance").agg(D=("D", "first"), sec=("wall_seconds", "mean"),
                                      eps=("evals_per_second", "mean"),
                                      fes=("fes_used", "mean"), runs=("seed", "count"))
        m["timing"] = dict(table=g.sort_index(), one_each_min=g.sec.sum() / 60,
                           pass_hours=g.sec.sum() * int(g.runs.max()) / 3600,
                           seeds=int(g.runs.max()), n=len(g),
                           fastest=g.sec.idxmin(), slowest=g.sec.idxmax(),
                           eps_small=g.eps.get("ta01"), eps_large=g.eps.get("ta71"))
    tc = _read("timing_contention.csv")
    if tc is not None:
        c = tc.pivot_table(index="instance", columns="jobs", values="wall_seconds",
                           aggfunc="mean")
        if 1 in c.columns and 8 in c.columns:
            c = c.dropna()
            r = c[8] / c[1]
            m["contention"] = dict(table=c, ratio=r, mean=r.mean(), lo=r.min(), hi=r.max(),
                                   jobs=8, throughput=8 / r.mean())

    # ---- ATSA vs DTSA runtime (D011) -----------------------------------------------
    # Both --jobs 1, both 3 seeds/instance, both elapsed. Ratio > 1 => DTSA-core slower
    # per run. This is NOT an algorithm comparison: DTSA-core's loop is plain NumPy,
    # ATSA's is njit-compiled. The fair basis is the identical MaxFEs = D*1000 budget.
    at = _read("atsa_timing_serial.csv")
    if at is not None and ts is not None:
        a1 = at[at.jobs == 1].groupby("instance").agg(D=("D", "first"),
                                                       atsa=("wall_seconds", "mean"))
        d1 = ts[ts.jobs == 1].groupby("instance")["wall_seconds"].mean().rename("dtsa")
        j = a1.join(d1, how="inner").dropna().sort_index()
        if len(j):
            rows, ratios = [], []
            for i in j.index:
                av, dv = float(j.loc[i, "atsa"]), float(j.loc[i, "dtsa"])
                ratios.append(dv / av)
                rows.append(dict(instance=i, D=int(j.loc[i, "D"]), atsa=av, dtsa=dv,
                                 ratio=dv / av, faster=("ATSA" if av < dv else "DTSA")))
            m["runtime"] = dict(
                rows=rows, n=len(rows), mean_ratio=sum(ratios) / len(ratios),
                ratio_ta71=(j.loc["ta71", "dtsa"] / j.loc["ta71", "atsa"]
                            if "ta71" in j.index else float("nan")),
                atsa_faster_on=sum(1 for r in rows if r["faster"] == "ATSA"),
                seeds=int(ts[ts.jobs == 1].groupby("instance").size().max()))

    ac = _read("atsa_timing_contention.csv")
    if ac is not None:
        piv = ac[ac.seed == 0].pivot_table(index="instance", columns="jobs",
                                            values="wall_seconds", aggfunc="mean")
        if 1 in piv.columns:
            levels = [jb for jb in (1, 8, 24) if jb in piv.columns]
            slow = {jb: (piv[jb] / piv[1]).dropna() for jb in levels if jb != 1}
            m["atsa_contention"] = dict(
                table=piv[levels], levels=levels, top=max(levels),
                mean={jb: r.mean() for jb, r in slow.items()},
                lo={jb: r.min() for jb, r in slow.items()},
                hi={jb: r.max() for jb, r in slow.items()})

    # ---- the equivariance trial count, from the test's own constants ----------------
    try:
        from tests.test_operators import EQUIVARIANCE_INSTANCES, EQUIVARIANCE_TRIALS
        import operators as _ops
        m["equivariance_trials"] = (EQUIVARIANCE_TRIALS * len(_ops.OPERATOR_ORDER)
                                    * len(EQUIVARIANCE_INSTANCES))
    except Exception:
        m["equivariance_trials"] = None
    return m


if __name__ == "__main__":
    d = M()
    for k in ("sampler", "complete_groups", "equivariance_trials"):
        print(f"{k:24s} {d.get(k)}")
    for ns, v in d.get("jssp", {}).items():
        print(f"{ns:24s} ours {v['adv_ours']}   paper {v['adv_paper']}   wins {v['wins']}/{v['n']}")
    if "mwkr" in d:
        print(f"{'mwkr ablation':24s} seeded {d['mwkr']['gap_seeded']}  "
              f"unseeded {d['mwkr']['gap_unseeded']}  retained {d['mwkr']['retained']:.0f}%")
    if "b4" in d:
        print(f"{'b4':24s} gain {d['b4']['gain']:+.2f}%  before {d['b4']['gap_before']}  "
              f"after {d['b4']['gap_after']}  closed {d['b4']['closed']:.0f}%")
