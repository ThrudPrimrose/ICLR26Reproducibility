"""Optimized tsvc_2 s255 for the Python arm.

The reference recurrence:
    x = b[N-1]; y = b[N-2]
    for i in 0..N-1:
        a[i] = (b[i] + x + y) * 0.333
        y = x; x = b[i]
unrolls to
    a[0]   = (b[0] + b[N-1] + b[N-2]) * 0.333
    a[1]   = (b[1] + b[0]   + b[N-1]) * 0.333
    a[i>=2]= (b[i] + b[i-1] + b[i-2]) * 0.333
with the same left-to-right addition order, so results are bit-identical
to the reference. The body is an independent 3-wide gather per element,
fully vectorizable and parallelizable.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=False)
def _s255_par(a, b, n):
    for i in prange(2, n):
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333
    a[0] = (b[0] + b[n - 1] + b[n - 2]) * 0.333
    a[1] = (b[1] + b[0] + b[n - 1]) * 0.333


@njit(parallel=False, fastmath=False)
def _s255_ser(a, b, n):
    for i in range(2, n):
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333
    a[0] = (b[0] + b[n - 1] + b[n - 2]) * 0.333
    a[1] = (b[1] + b[0] + b[n - 1]) * 0.333


def s255(a, b, LEN_1D):
    n = int(LEN_1D)
    if n >= 2:
        if n >= 2 ** 19:
            _s255_par(a, b, n)
        else:
            _s255_ser(a, b, n)
    return None


# Warm up both compiled paths at import time so first call pays nothing.
_w = np.zeros(4096)
_wb = np.zeros(4096)
_s255_ser(_w, _wb, 4096)
_s255_par(_w, _wb, 4096)
