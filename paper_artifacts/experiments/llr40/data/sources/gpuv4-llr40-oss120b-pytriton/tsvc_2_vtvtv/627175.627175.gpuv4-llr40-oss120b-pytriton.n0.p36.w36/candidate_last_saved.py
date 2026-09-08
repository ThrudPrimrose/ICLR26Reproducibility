"""Optimized implementation of the TSVC_2 ``vtvtv`` kernel using Numba parallelism.

The kernel multiplies three 1‑D arrays element‑wise and stores the result back
into ``a`` in‑place:

    a[i] = a[i] * b[i] * c[i]

A Numba ``@njit`` function with ``parallel=True`` and ``prange`` enables the
operation to be multithreaded, which gives a substantial speed‑up for the larger
input sizes used in the fuzzed benchmark preset.  To avoid counting JIT
compilation time in the timed run, we perform a tiny warm‑up call at module import
time.  The function returns ``None`` to indicate an in‑place update, as required
by the benchmark harness.
"""

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _vtvtv_impl(a: np.ndarray, b: np.ndarray, c: np.ndarray, LEN_1D: int) -> None:
    for i in prange(LEN_1D):
        a[i] = a[i] * b[i] * c[i]

# Warm‑up compilation (using double precision, matching the reference C kernel).
_dummy_a = np.empty(1, dtype=np.float64)
_dummy_b = np.empty(1, dtype=np.float64)
_dummy_c = np.empty(1, dtype=np.float64)
_vtvtv_impl(_dummy_a, _dummy_b, _dummy_c, 1)

def vtvtv(a: np.ndarray, b: np.ndarray, c: np.ndarray, LEN_1D: int) -> None:
    _vtvtv_impl(a, b, c, LEN_1D)
    return None
