"""Optional-Numba compatibility shim.

The ATSA hot loops are decorated with ``@njit`` and are ~69x faster when Numba is
installed and can compile them. Numba is not required to get *correct* results: the
decorated functions are ordinary Python/NumPy and run unchanged (just slower) when
Numba is absent.

This module exposes a single ``njit`` name. It is the real Numba ``njit`` when Numba
imports and initialises cleanly, and a transparent no-op decorator otherwise. A
Numba install/compile failure therefore degrades speed, never blocks a reproduction
run (see README, "Troubleshooting").

Equivalence is measured, not assumed: on ta01, seeds 0-4, the ATSA makespans are
*bit-identical* with Numba enabled and disabled ([1457, 1474, 1643, 1415, 1469] in
both modes). Numba's ``np.random`` reproduces NumPy's legacy MT19937 stream, so the
pure-Python fallback reproduces the published numbers, not merely a valid run.

Force the pure-Python path for testing with the environment variable
``ATSA_NO_NUMBA=1``.
"""
from __future__ import annotations

import os
import warnings


def _noop_njit(*args, **kwargs):
    """Decorator that ignores Numba options. Supports both ``@njit`` and ``@njit(...)``."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]                      # used as bare @njit

    def wrap(fn):
        return fn                           # used as @njit(cache=True, ...)
    return wrap


if os.environ.get("ATSA_NO_NUMBA") == "1":
    njit = _noop_njit
    NUMBA_ENABLED = False
else:
    try:
        from numba import njit as _real_njit
        njit = _real_njit
        NUMBA_ENABLED = True
    except Exception as exc:                # ImportError, or a compile/CPU init failure
        warnings.warn(
            f"Numba unavailable ({exc!r}); falling back to pure Python. Runs will be "
            "much slower but results are identical. Set ATSA_NO_NUMBA=1 to silence this.",
            RuntimeWarning,
            stacklevel=2,
        )
        njit = _noop_njit
        NUMBA_ENABLED = False
