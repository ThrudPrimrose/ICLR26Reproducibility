"""TSVC tsvc_2 s255 optimized (python arm).

The reference recurrence
    x = b[n-1]; y = b[n-2]
    for i: a[i] = (b[i] + x + y) * 0.333; y = x; x = b[i]
carries only b[i-1] and b[i-2] forward (never any a), so it is exactly
    a[i] = ((b[i] + b[(i-1) mod n]) + b[(i-2) mod n]) * 0.333
and parallelizable with no change in floating-point rounding (bit-exact).
"""
import numpy as np
from numba import njit, prange
from numba.np.ufunc import get_thread_id


@njit(parallel=False, fastmath=False)
def _ser(a, b, n):
    x = b[n - 1]
    y = b[n - 2]
    for i in range(n):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]


@njit(parallel=True, fastmath=False)
def _par(a, b, n):
    m = (n - 2) // 8
    for j in prange(m):
        i = 2 + 8 * j
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333
        a[i + 1] = (b[i + 1] + b[i] + b[i - 1]) * 0.333
        a[i + 2] = (b[i + 2] + b[i + 1] + b[i]) * 0.333
        a[i + 3] = (b[i + 3] + b[i + 2] + b[i + 1]) * 0.333
        a[i + 4] = (b[i + 4] + b[i + 3] + b[i + 2]) * 0.333
        a[i + 5] = (b[i + 5] + b[i + 4] + b[i + 3]) * 0.333
        a[i + 6] = (b[i + 6] + b[i + 5] + b[i + 4]) * 0.333
        a[i + 7] = (b[i + 7] + b[i + 6] + b[i + 5]) * 0.333
    if get_thread_id() == 0:
        a[0] = (b[0] + b[n - 1] + b[n - 2]) * 0.333
        a[1] = (b[1] + b[0] + b[n - 1]) * 0.333
        for i in range(2 + 8 * m, n):
            a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333


_PAR_MIN = 1 << 18


def s255(a, b, LEN_1D):
    n = LEN_1D
    if n >= _PAR_MIN:
        _par(a, b, n)
    else:
        _ser(a, b, n)
    return None


# Warm up / compile at import time so the timed call is pure execution.
_n = 64
_a = np.zeros(_n)
_b = np.zeros(_n)
_ser(_a, _b, _n)
_par(_a, _b, _n)
_n2 = _PAR_MIN + 7
_a2 = np.zeros(_n2)
_b2 = np.zeros(_n2)
_par(_a2, _b2, _n2)
del _a, _b, _a2, _b2
