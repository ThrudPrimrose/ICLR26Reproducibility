# TSVC s316: min-reduction over a 1D float64 array -> result[0]
# Parallel chunked min via numba prange; compiled + warmed at import time.
import os

try:
    _T = max(1, len(os.sched_getaffinity(0)))
except Exception:
    _T = os.cpu_count() or 24
os.environ["NUMBA_NUM_THREADS"] = str(_T)

import numpy as np
import numba  # noqa: F401  (import AFTER setting NUMBA_NUM_THREADS)
from numba import njit, prange


@njit(parallel=True, fastmath=True, boundscheck=False)
def _min_parts(a, p, T):
    n = a.shape[0]
    for t in prange(T):
        lo = (t * n) // T
        hi = ((t + 1) * n) // T
        x = np.inf
        for i in range(lo, hi):
            v = a[i]
            if v < x:
                x = v
        p[t] = x


@njit(fastmath=True, boundscheck=False)
def _min_serial(a):
    n = a.shape[0]
    x = a[0]
    for i in range(1, n):
        if a[i] < x:
            x = a[i]
    return x


_parts = np.empty(_T, dtype=np.float64)


def s316(a, result, LEN_1D):
    n = a.shape[0]
    if n < _T * 1024:
        result[0] = _min_serial(a)
    else:
        _min_parts(a, _parts, _T)
        result[0] = _parts.min()
    return None


# Pre-warm: compile both kernels and start the thread pool before timing.
_dummy = np.linspace(1.0, 0.0, 1 << 20)
_min_parts(_dummy, _parts, _T)
_min_serial(_dummy)
