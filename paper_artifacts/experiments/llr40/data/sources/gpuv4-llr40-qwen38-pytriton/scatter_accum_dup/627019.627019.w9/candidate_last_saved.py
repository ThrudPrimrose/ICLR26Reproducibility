import numpy as np
import numba
from numba import njit, prange


@njit(parallel=True, fastmath=False)
def _scatter(bins, src, ip, scratch, n, T):
    SL = (n + T - 1) // T
    for t in prange(T):
        lo = t * SL
        hi = lo + SL
        if hi > n:
            hi = n
        for k in range(lo, hi):
            scratch[k] = 0.0
        for i in range(n):
            k = ip[i]
            if k >= lo and k < hi:
                scratch[k] += src[i]


@njit(parallel=True, fastmath=False)
def _merge(bins, scratch, n):
    for k in prange(n):
        bins[k] += scratch[k]


def scatter_accum_dup(bins, src, ip, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    T = numba.get_num_threads()
    if n < 200000:
        T2 = n // 8000
        if T2 < 1:
            T2 = 1
        if T2 < T:
            T = T2
    scratch = np.empty(n, dtype=np.float64)
    _scatter(bins, src, ip, scratch, n, T)
    _merge(bins, scratch, n)
    return None


# Warm up / compile at import time so the first timed call is fast.
def _warmup():
    n = 1 << 12
    b = np.zeros(n)
    s = np.ones(n)
    i = np.arange(n, dtype=np.int32)
    sc = np.empty(n)
    _scatter(b, s, i, sc, n, 2)
    _merge(b, sc, n)


_warmup()
