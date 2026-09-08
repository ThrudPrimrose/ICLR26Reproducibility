# TSVC quasi_affine_reduce_odd: out[0] = sum(a[i] for i in range(1, LEN_1D, 2))
#
# Memory-bandwidth-bound strided reduction over a ~3.5-4.2 GB fp64 array.  The odd
# elements are interleaved with the even ones, so every 64-byte cache line is read
# (full-array traffic) and the per-core job is just to add the odd lanes.  We hit the
# socket's memory ceiling (~200 GB/s) with a numba prange reduction, which the LLVM
# backend vectorizes the stride-2 fp64 load of (with fastmath).  The kernel is compiled
# and warmed at import time -- outside the timed section -- so every timed call runs
# native code.
import os

import numpy as np
import numba as nb
from numba import prange


def _affinity_count():
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(64, n))


_NT = _affinity_count()
nb.set_num_threads(_NT)


@nb.njit(parallel=True, fastmath=True, cache=False)
def _odd_sum(a, m):
    # a[1], a[3], ..., a[2*m-1]  (m = number of odd indices in [1, LEN_1D))
    t = 0.0
    for j in prange(m):
        t += a[2 * j + 1]
    return t


# Compile + warm at import (before the clock).  Small dummy so the real call is native.
_warm = np.zeros(4096, dtype=np.float64)
_odd_sum(_warm, 2048)
del _warm

_SMALL = 1_000_000  # below this, a single vectorized numpy reduce beats prange startup


def quasi_affine_reduce_odd(a, out, LEN_1D):
    n = int(LEN_1D) // 2  # count of odd i with 1 <= i < LEN_1D
    if n <= 0:
        out[0] = 0.0
        return None
    if n < _SMALL:
        out[0] = np.add.reduce(a[1 : 1 + 2 * n : 2])
        return None
    out[0] = _odd_sum(a, n)
    return None
