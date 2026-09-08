"""Optimized tsvc_2 s152: b[i] = d[i]*e[i]; a[i] += b[i]*c[i]  (in-place on a, b).

Memory-bandwidth-bound elementwise kernel. Fused into one loop (6 streams, the
minimum: 4 reads a/d/e/c + 2 writes b/a) and parallelized with numba prange for
large n; serial numba path for small n where thread-pool dispatch would dominate.
Everything is precompiled and warmed at import time (not timed).
"""
import numpy as np
import numba
from numba import njit, prange


@njit(parallel=False, cache=False, nogil=True)
def _serial(a, b, c, d, e):
    n = a.shape[0]
    for i in range(n):
        t = d[i] * e[i]
        b[i] = t
        a[i] = a[i] + t * c[i]


@njit(parallel=True, cache=False, nogil=True)
def _parallel(a, b, c, d, e):
    n = a.shape[0]
    for i in prange(n):
        t = d[i] * e[i]
        b[i] = t
        a[i] = a[i] + t * c[i]


# crossover: below this, thread-pool dispatch overhead > kernel work
_THR = 200000


def s152(a, b, c, d, e, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    if n < _THR:
        _serial(a, b, c, d, e)
    else:
        _parallel(a, b, c, d, e)
    return None


# ---- import-time precompile + warm (runs once, before the timer) ----
_d = np.zeros(1024, dtype=np.float64)
_bc = np.zeros(1024, dtype=np.float64)
_serial(_d, _bc.copy(), _d, _d, _d)
_parallel(_d, _bc.copy(), _d, _d, _d)
