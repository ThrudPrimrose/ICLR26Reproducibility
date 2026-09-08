"""Optimized TSVC ext_war_unit: a[i] = a[i+1] + b[i] for i in 0..LEN_1D-2.

The loop's only cross-iteration effect is an anti-dependence (thread i reads
a[i+1], thread i+1 writes a[i+1]); the final value a[i] always uses the
ORIGINAL a[i+1].  So the result is a shifted read + add, safely parallelized
with a scratch buffer:
    tmp[i] = a[i+1] + b[i]     (parallel)
    a[i]   = tmp[i]            (parallel)
The numba kernels are compiled + warmed at import time (untimed).
"""
import numpy as np
import numba
from numba import njit, prange


@njit(parallel=True, fastmath=True, nogil=True)
def _war_unit2(a, b, tmp, m):
    for i in prange(m):
        tmp[i] = a[i + 1] + b[i]
    for i in prange(m):
        a[i] = tmp[i]


_tmp = None


def ext_war_unit(a, b, LEN_1D):
    global _tmp
    m = LEN_1D - 1
    if m <= 0:
        return
    t = _tmp
    if t is None or t.dtype != a.dtype or t.shape[0] < m:
        t = np.empty(m, dtype=a.dtype)
        _tmp = t
    _war_unit2(a, b, t, m)


# ---- import-time warmup (untimed): compile all dtype variants we may see ----
def _warm():
    for dt in (np.float64, np.float32, np.int64):
        n = 1 << 20
        a = np.arange(n, dtype=dt)
        bb = np.arange(n, dtype=dt)
        ext_war_unit(a, bb, n)
_warm()
del _warm
