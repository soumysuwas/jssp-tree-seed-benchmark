"""
One source of truth for the 40-instance paper comparison scope.

Two places name that set: PAPER_INSTANCES in src/atsa_jssp/instance.py (used by the code)
and data/paper_scope_40.txt (a human-readable list, derived from the keys of
src/atsa_jssp/paper_table5.py). This test fails if they ever drift apart.
"""
from __future__ import annotations

import pathlib

from atsa_jssp.instance import PAPER_INSTANCES

REPO = pathlib.Path(__file__).resolve().parents[1]
SCOPE_FILE = REPO / "data" / "paper_scope_40.txt"


def _scope_from_file() -> list[str]:
    lines = SCOPE_FILE.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def test_scope_is_forty_instances():
    assert len(PAPER_INSTANCES) == 40
    assert len(_scope_from_file()) == 40


def test_scope_file_matches_paper_instances():
    assert set(_scope_from_file()) == set(PAPER_INSTANCES)
