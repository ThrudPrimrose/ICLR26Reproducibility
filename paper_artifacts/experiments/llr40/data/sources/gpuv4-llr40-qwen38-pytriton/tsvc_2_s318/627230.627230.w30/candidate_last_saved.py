"""TSVC tsvc_2 kernel s318: result[0] = max_i |a[i*inc]| + float(first index attaining it).

Single-pass parallel first-argmax over the strided array: the range is split into T
segments processed in parallel; each segment keeps its local (max |value|, first local
index), then the T pairs are combined serially keeping the earliest index on ties.
Numba pre-compiled (and warmed) at import time so the timed call is pure native code.
"""
import numpy as np
import numba as nb
from numba import njit, prange


@njit()
def _serial_first_argmax(a, n, inc):
    maxv = abs(a[0])
    index = 0
    k = inc
    for i in range(1, n):
        v = abs(a[k])
        if v > maxv:
            index = i
            maxv = v
        k += inc
    return maxv + float(index)


@njit(parallel=True)
def _parallel_first_argmax(a, n, inc, T):
    seg = (n + T - 1) // T
    sv = np.empty(T)
    si = np.empty(T, dtype=np.int64)
    for s in prange(T):
        lo = s * seg
        hi = lo + seg
        if hi > n:
            hi = n
        k = lo * inc
        mv = abs(a[k])
        mi = lo
        for i in range(lo + 1, hi):
            k += inc
            v = abs(a[k])
            if v > mv:
                mi = i
                mv = v
        sv[s] = mv
        si[s] = mi
    bv = sv[0]
    bi = si[0]
    for s in range(1, T):
        if sv[s] > bv:
            bv = sv[s]
            bi = si[s]
    return bv + float(bi)


_T = max(1, int(nb.get_num_threads()))


def s318(a, result, inc, LEN_1D):
    n = int(LEN_1D)
    inc = int(inc)
    if n <= 1:
        result[0] = abs(float(a[0])) + 0.0 if n == 1 else 0.0
        return None
    T = _T
    if T > n:
        T = n
    result[0] = _parallel_first_argmax(a, n, inc, T)
    return None


# Warm / pre-compile at import time: the first launch would otherwise pay LLVM
# compilation inside the timed region.
_dummy = np.arange(2 * 1024 * 1024 + 1, dtype=np.float64)[::1]
_dummy_r = np.empty(1)
s318(_dummy, _dummy_r, 1, len(_dummy))
s318(_dummy, _dummy_r, 1, 1)
s318(_dummy, _dummy_r, 1, 2)
del _dummy, _dummy_r
