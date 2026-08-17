"""Command-line interface for the ATSA reproduction.

    # one instance, 20 seeded runs
    python -m atsa_jssp.cli run --instance ta01 --runs 20

    # all 40 paper instances (ta71-75 run last)
    python -m atsa_jssp.cli run --all --runs 20

    # basic TSA baseline instead of ATSA
    python -m atsa_jssp.cli run --instance ta01 --algorithm TSA

By default the number of parallel workers equals the machine's logical core count;
override with --jobs. Results are written to a CSV (path printed on completion).
"""
from __future__ import annotations
import os

# MUST be set before numba is imported anywhere, so the JIT does not spawn its own
# thread pool on top of the process pool.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pathlib
from typing import Optional

import typer
from rich.console import Console

from atsa_jssp.atsa import Config
from atsa_jssp.experiment import ROOT, run_experiment, summarise
from atsa_jssp.instance import PAPER_INSTANCES

app = typer.Typer(add_completion=False, help="ATSA reproduction (Sahman 2022) on the Taillard JSSP.")
console = Console()


@app.callback()
def _main() -> None:
    """Keep `run` as an explicit subcommand (so `cli run ...` works as documented)."""


@app.command()
def run(
    instance: Optional[str] = typer.Option(None, help="single instance, e.g. ta01"),
    all: bool = typer.Option(False, "--all", help="all 40 paper instances (ta71-75 last)"),
    runs: int = typer.Option(20, help="independent runs; seeds 0..runs-1"),
    jobs: int = typer.Option(os.cpu_count() or 1, help="parallel workers (default: logical cores)"),
    st_sense: str = typer.Option("rand_lt_st", help="rand_lt_st (pseudocode) | st_lt_rand (prose)"),
    operator_space: str = typer.Option("continuous", help="continuous | permutation (INERT - see atsa.py)"),
    branch_granularity: str = typer.Option("seed", help="seed (Alg.2) | dimension (Alg.1)"),
    strict_fe_cap: bool = typer.Option(False, help="True = break mid-loop at MaxFEs (non-literal)"),
    fe_multiplier: int = typer.Option(1000, help="MaxFEs = D * this. Table 4 says 1000."),
    algorithm: str = typer.Option("ATSA", help="ATSA (Algorithm 2) | TSA (Algorithm 1 baseline)"),
    out: Optional[pathlib.Path] = typer.Option(None, help="CSV output path"),
) -> None:
    """Run ATSA or plain TSA. Default config = the most literal reading of the paper."""
    if all:
        names = PAPER_INSTANCES
    elif instance:
        names = [instance]
    else:
        raise typer.BadParameter("give --instance or --all")

    cfg = Config(st_sense=st_sense, operator_space=operator_space,
                 branch_granularity=branch_granularity, strict_fe_cap=strict_fe_cap,
                 fe_multiplier=fe_multiplier)
    out = out or ROOT / f"results/{algorithm.lower()}_{'full' if all else names[0]}.csv"

    console.print(f"[bold]{algorithm.upper()}[/bold]  {len(names)} instance(s) x {runs} runs  "
                  f"jobs={jobs}\n  {cfg.st_sense} / {cfg.branch_granularity} / "
                  f"{cfg.operator_space} / strict_fe_cap={cfg.strict_fe_cap}")
    df = run_experiment(names, runs=runs, cfg=cfg, jobs=jobs, out=out, algorithm=algorithm)
    console.print(summarise(df).to_string(index=False))
    console.print(f"\n[green]wrote[/green] {out}  ({len(df)} rows)")


if __name__ == "__main__":                              # required for multiprocessing on Windows/macOS
    app()
