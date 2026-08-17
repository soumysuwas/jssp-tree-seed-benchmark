#!/usr/bin/env python3
"""
JSSP port smoke test -- ta01, 2 seeds. THE ONLY EXECUTION PERMITTED IN PART C.

Confirms three things and nothing else: the D2a round trip holds for every instance, the port
runs, and the FE identity holds. It is NOT a result. No makespan here is compared to anything
published, and the 40-instance run stays BLOCKED until Gate 1 passes (the DTSA adaptation notes D5).

Usage:  uv run python dtsa/smoke_test_jssp.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dtsa_jssp import (JSSPConfig, check_roundtrip_all_instances, mwkr_sequence,  # noqa: E402
                       rk_from_sequence, run_one, system_config, _schedule)
from atsa_jssp.decoder import evaluate_fast                                        # noqa: E402
from atsa_jssp.instance import load_ta                                             # noqa: E402


def main() -> None:
    print("--- D2a: sequence -> random key -> sequence, EVERY Taillard instance -------------")
    rt = check_roundtrip_all_instances()
    bad = [k for k, ok in rt.items() if not ok]
    print(f"  {sum(rt.values())}/{len(rt)} instances round-trip exactly"
          f"{'' if not bad else '   FAILED: ' + str(bad)}")
    if bad:
        raise SystemExit("D2a failed -- MWKR seeding cannot be trusted. STOP.")

    print("\n--- D2: MWKR seed vs the decoder ------------------------------------------------")
    for name in ["ta01", "ta11", "ta21"]:
        inst = load_ta(name)
        route, ptime = inst.arrays()
        seq = mwkr_sequence(inst)
        mwkr_cmax, _, _, _ = _schedule(seq, inst)
        seeded = int(evaluate_fast(rk_from_sequence(seq, inst.n), route, ptime, inst.n, inst.m))
        print(f"  {name}: MWKR's own makespan {mwkr_cmax:>6}   decoded from the seeded tree "
              f"{seeded:>6}   {'EQUAL' if mwkr_cmax == seeded else 'MISMATCH'}")

    print("\n--- ta01, 2 seeds, DTSA-core, N = D = 225, MaxFEs = D*1000 ----------------------")
    for cfg_name, jcfg in [("DTSA-core", JSSPConfig(use_local_search=False)),
                           ("DTSA+LS", JSSPConfig(use_local_search=True))]:
        for seed in ([0, 1] if cfg_name == "DTSA-core" else [0]):
            r = run_one("ta01", seed, jcfg)
            ident = r["fes_used"] == r["N"] + 6 * r["N"] * r["iters"]
            print(f"  {cfg_name:<10} seed {seed}: cmax {r['cmax']:>5}  "
                  f"(pre-LS {r['pre_local_search_cmax']:>5})  "
                  f"fes {r['fes_used']:>7} = {r['N']} + 6*{r['N']}*{r['iters']}  "
                  f"[{'OK' if ident else 'FAIL'}]  "
                  f"LS evals {r['local_search_evals']:>5}  {r['wall_s']:>6.1f}s")

    print("\n--- system_config (written alongside every result set) --------------------------")
    print(json.dumps(system_config(), indent=2))
    print("\nSmoke test only. Not a result. The 40-instance run is blocked on Gate 1.")


if __name__ == "__main__":
    main()
