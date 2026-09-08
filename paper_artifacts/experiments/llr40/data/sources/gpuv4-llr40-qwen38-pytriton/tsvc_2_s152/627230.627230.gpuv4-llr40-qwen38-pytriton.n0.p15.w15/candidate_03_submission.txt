import os
import numpy as np
import numba
from numba import njit, prange

numba.set_num_threads(max(1, len(os.sched_getaffinity(0))))


@njit(parallel=True)
def _k(a, b, c, d, e):
    n = a.shape[0]
    for i in prange(n):
        b[i] = d[i] * e[i]
        a[i] += b[i] * c[i]


def s152(a, b, c, d, e, LEN_1D):
    _k(a, b, c, d, e)
    return None


# Warm the JIT (compile + thread pool) at import time, before the timer starts.
_w = np.zeros(1 << 20)
_k(_w, _w.copy(), _w.copy(), _w.copy(), _w.copy())
