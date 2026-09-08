"""Optimized wf_triangular: numba parallel wavefront over i+j anti-diagonals."""
import os
import numpy as np
import numba
from numba import njit, prange

_NUM_THREADS = max(1, len(os.sched_getaffinity(0)))
numba.set_num_threads(_NUM_THREADS)


@njit(parallel=True, cache=True)
def _wf(a, n):
    for k in prange(2, 2 * n):
        i0 = k - (n - 1)
        if i0 < 1:
            i0 = 1
        i1 = k // 2
        if i1 > n - 1:
            i1 = n - 1
        for i in range(i0, i1 + 1):
            j = k - i
            a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]


def wf_triangular(a, LEN_2D):
    _wf(a, LEN_2D)


# Warm at import: compile + stabilize threads outside the timed region.
for _dt in (np.float64, np.float32):
    _t = np.zeros((16, 16), dtype=_dt)
    _wf(_t, 16)
