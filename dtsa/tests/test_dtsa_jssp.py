"""
The job-shop port's plumbing -- the DTSA adaptation notes D1-D4.

NOT a validation of DTSA on job shop. Nothing here compares a makespan to anything published,
and nothing may, until Gate 1 passes (D5). These tests check that the port is internally
consistent and that it agrees with the VERIFIED ATSA decoder wherever the two overlap.
"""
from __future__ import annotations

import numpy as np
import pytest

from atsa_jssp.decoder import evaluate_fast, rk_to_job_sequence
from atsa_jssp.instance import PAPER_INSTANCES, load_ta
from dtsa_jssp import (JSSPConfig, _schedule, check_roundtrip_all_instances, mwkr_sequence,
                       n5_local_search, rk_from_sequence, run_one, system_config)

SMALL = ["ta01", "ta11", "ta21"]


# ==============================================================================================
# D2a -- the sequence <-> random-key round trip, for EVERY instance
# ==============================================================================================
def test_d2a_roundtrip_holds_for_every_taillard_instance():
    """
    the DTSA adaptation notes D2a. MWKR produces a job sequence; `rk_from_sequence` must produce a
    random-key vector that decodes back to exactly that sequence. If this fails anywhere, the
    seeded tree is silently a different schedule from the one MWKR built, and the D2 decision is
    unusable.
    """
    results = check_roundtrip_all_instances(PAPER_INSTANCES)
    failed = [k for k, ok in results.items() if not ok]
    assert not failed, f"round trip failed on {failed}"
    assert len(results) == 40


@pytest.mark.parametrize("name", SMALL)
def test_roundtrip_holds_for_arbitrary_sequences_not_just_mwkr(name):
    inst = load_ta(name)
    rng = np.random.default_rng(0)
    for _ in range(50):
        seq = rk_to_job_sequence(rng.uniform(-5, 5, inst.D), inst.n)
        back = rk_to_job_sequence(rk_from_sequence(seq, inst.n), inst.n)
        assert np.array_equal(back, seq)


@pytest.mark.parametrize("name", SMALL)
def test_rk_from_sequence_stays_inside_the_atsa_range(name):
    inst = load_ta(name)
    x = rk_from_sequence(mwkr_sequence(inst), inst.n)
    assert x.min() >= -5.0 and x.max() <= 5.0


# ==============================================================================================
# D2 -- MWKR
# ==============================================================================================
@pytest.mark.parametrize("name", SMALL)
def test_mwkr_emits_a_valid_job_sequence(name):
    inst = load_ta(name)
    seq = mwkr_sequence(inst)
    assert seq.shape == (inst.D,)
    counts = np.bincount(seq, minlength=inst.n)
    assert (counts == inst.m).all(), "each job must appear exactly m times"


@pytest.mark.parametrize("name", SMALL)
def test_mwkr_seed_decodes_to_the_makespan_mwkr_actually_built(name):
    """
    The seeded tree's decoded makespan must EQUAL the makespan of MWKR's own sequence -- not
    merely be no worse. The round trip is exact, so anything else means the seeding lost the
    schedule it was supposed to inject.
    """
    inst = load_ta(name)
    route, ptime = inst.arrays()
    seq = mwkr_sequence(inst)
    mwkr_cmax, _, _, _ = _schedule(seq, inst)
    seeded_cmax = int(evaluate_fast(rk_from_sequence(seq, inst.n), route, ptime, inst.n, inst.m))
    assert seeded_cmax == mwkr_cmax


@pytest.mark.parametrize("name", SMALL)
def test_mwkr_beats_a_typical_random_key_vector(name):
    """Not a correctness requirement, but if MWKR were no better than random the D2 decision
    would be inert and worth revisiting. Recorded as a property, checked loosely."""
    inst = load_ta(name)
    route, ptime = inst.arrays()
    rng = np.random.default_rng(0)
    mwkr = int(evaluate_fast(rk_from_sequence(mwkr_sequence(inst), inst.n),
                             route, ptime, inst.n, inst.m))
    randoms = [int(evaluate_fast(rng.uniform(-5, 5, inst.D), route, ptime, inst.n, inst.m))
               for _ in range(30)]
    assert mwkr < np.median(randoms)


# ==============================================================================================
# The duplicated decode logic -- guarded, not trusted
# ==============================================================================================
@pytest.mark.parametrize("name", SMALL)
def test_local_schedule_matches_the_verified_decoder(name):
    """
    `dtsa_jssp._schedule` re-implements the semi-active rule because the local search needs
    machine orders and `src/atsa_jssp/decoder.py` is read-only. A second copy of decode logic is
    exactly what has bitten this project before, so it is checked against the verified one on
    random inputs rather than assumed.
    """
    inst = load_ta(name)
    route, ptime = inst.arrays()
    rng = np.random.default_rng(1)
    for _ in range(200):
        x = rng.uniform(-5, 5, inst.D)
        seq = rk_to_job_sequence(x, inst.n)
        mine, _, _, _ = _schedule(seq, inst)
        theirs = int(evaluate_fast(x, route, ptime, inst.n, inst.m))
        assert mine == theirs


@pytest.mark.parametrize("name", SMALL)
def test_machine_orders_are_consistent(name):
    inst = load_ta(name)
    rng = np.random.default_rng(2)
    seq = rk_to_job_sequence(rng.uniform(-5, 5, inst.D), inst.n)
    _, starts, ends, machine_order = _schedule(seq, inst)
    assert sum(len(o) for o in machine_order) == inst.D
    for mach, ops_on_m in enumerate(machine_order):
        assert len(ops_on_m) == inst.n, "every job visits every machine exactly once"
        for (j1, k1), (j2, k2) in zip(ops_on_m, ops_on_m[1:]):
            assert ends[j1, k1] <= starts[j2, k2], "machine overlap"


# ==============================================================================================
# D3 -- N5 local search
# ==============================================================================================
@pytest.mark.parametrize("name", ["ta01"])
def test_n5_never_worsens_and_returns_a_decodable_vector(name):
    inst = load_ta(name)
    route, ptime = inst.arrays()
    rng = np.random.default_rng(3)
    for _ in range(5):
        x = rng.uniform(-5, 5, inst.D)
        before = int(evaluate_fast(x, route, ptime, inst.n, inst.m))
        x2, after, evals, moves = n5_local_search(x, inst)
        assert after <= before
        assert evals > 0
        assert int(evaluate_fast(x2, route, ptime, inst.n, inst.m)) == after
        if moves > 0:
            assert after < before


def test_n5_actually_improves_a_random_solution():
    """If the neighbourhood never fired, the D3 alternative would be worthless and we would want
    to know before shipping DTSA+LS."""
    inst = load_ta("ta01")
    route, ptime = inst.arrays()
    rng = np.random.default_rng(4)
    improved = 0
    for _ in range(5):
        x = rng.uniform(-5, 5, inst.D)
        before = int(evaluate_fast(x, route, ptime, inst.n, inst.m))
        _, after, _, _ = n5_local_search(x, inst)
        improved += after < before
    assert improved >= 4


# ==============================================================================================
# D4 -- accounting and schema
# ==============================================================================================
def test_fe_identity_and_row_schema_on_a_short_ta01_run():
    """A deliberately tiny budget: this checks accounting and schema, NOT quality."""
    from dtsa_jssp import CSV_COLUMNS
    jcfg = JSSPConfig(n_trees=8, fe_multiplier=2, use_local_search=False)
    row = run_one("ta01", 0, jcfg)
    assert set(row) == set(CSV_COLUMNS)
    assert row["algorithm"] == "DTSA" and row["config"] == "DTSA-core"
    assert row["fes_used"] == row["N"] + 6 * row["N"] * row["iters"]
    assert row["fes_used"] >= row["max_fes"]
    assert row["fe_per_seed"] == 1.0
    assert row["n_seeds"] == row["fes_used"] - row["N"]
    assert row["local_search_evals"] == 0


def test_local_search_evaluations_are_never_folded_into_fes():
    """D3/U7 -- the entire fairness argument for the job-shop column rests on this."""
    jcfg = JSSPConfig(n_trees=8, fe_multiplier=2, use_local_search=True)
    row = run_one("ta01", 0, jcfg)
    assert row["config"] == "DTSA+LS"
    assert row["fes_used"] == row["N"] + 6 * row["N"] * row["iters"]   # untouched by LS
    assert row["local_search_evals"] > 0
    assert row["cmax"] <= row["pre_local_search_cmax"]


def test_n_setting_is_recorded_both_ways():
    assert JSSPConfig(n_trees=None).trees_for(225) == 225      # N = D, DTSA-literal
    assert JSSPConfig(n_trees=40).trees_for(225) == 40         # ATSA-matched
    assert JSSPConfig(n_trees=None).n_setting_slug() == "N_eq_D"
    assert JSSPConfig(n_trees=40).n_setting_slug() == "N40"


def test_system_config_records_what_the_lab_meeting_needs():
    sc = system_config()
    for key in ("os", "processor", "logical_cores", "ram_total_gb", "python", "numpy",
                "numba", "git_sha", "timestamp"):
        assert key in sc, key
    assert sc["logical_cores"] and sc["logical_cores"] > 0
