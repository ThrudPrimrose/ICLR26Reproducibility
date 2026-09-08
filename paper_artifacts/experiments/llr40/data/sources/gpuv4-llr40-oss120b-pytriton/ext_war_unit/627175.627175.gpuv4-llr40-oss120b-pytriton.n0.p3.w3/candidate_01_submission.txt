"""Optimized implementation of the TSVC ext_war_unit kernel using Numba.

The kernel computes, for i in [0, LEN_1D-2]:
    a[i] = a[i+1] + b[i]

It updates array ``a`` in-place. This version uses a Numba JIT‑compiled loop with
``fastmath`` enabled. The JIT compilation is performed once at import time so the
compile overhead does not affect the timed run.
"""

import numpy as np
import numba

# The compiled inner loop. ``fastmath`` enables aggressive floating‑point
# optimisations (e.g. fused multiply‑add, reassociation) which can give a modest
# speed boost over the default settings.
@numba.njit(fastmath=True, cache=True)
def _ext_war_unit_impl(a: np.ndarray, b: np.ndarray, LEN_1D: int) -> None:
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]

# Warm‑up the JIT compilation so that the first call during benchmarking is
# already compiled. Using a length of 1 triggers compilation without doing any
# real work.
_dummy_a = np.empty(1, dtype=np.float64)
_dummy_b = np.empty(1, dtype=np.float64)
_ext_war_unit_impl(_dummy_a, _dummy_b, 1)

def ext_war_unit(a: np.ndarray, b: np.ndarray, LEN_1D: int) -> None:
    """In‑place update of ``a`` according to the ``ext_war_unit`` pattern.

    Parameters
    ----------
    a : np.ndarray
        1‑D array of length ``LEN_1D``. Modified in‑place.
    b : np.ndarray
        1‑D array of length ``LEN_1D`` (only the first ``LEN_1D‑1`` elements are read).
    LEN_1D : int
        Logical length of the arrays. ``a`` and ``b`` may be larger, but only the
        first ``LEN_1D`` entries are considered.
    """
    if LEN_1D <= 1:
        return
    _ext_war_unit_impl(a, b, LEN_1D)
