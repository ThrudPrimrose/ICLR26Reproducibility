"""TSVC tsvc_2 s2710 -- bit-exact, multithreaded (numba prange) implementation."""
import numba
from numba import prange
import numpy as np


@numba.njit(fastmath=False)
def _s2710_serial(a, b, c, d, e, n, big, x0pos):
    for i in range(n):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * d[i]
            if big:
                c[i] = c[i] + d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            if x0pos:
                c[i] = a[i] + d[i] * d[i]
            else:
                c[i] = c[i] + e[i] * e[i]


@numba.njit(parallel=True, fastmath=False)
def _s2710_par(a, b, c, d, e, n, big, x0pos):
    for i in prange(n):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * d[i]
            if big:
                c[i] = c[i] + d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            if x0pos:
                c[i] = a[i] + d[i] * d[i]
            else:
                c[i] = c[i] + e[i] * e[i]


_SMALL = 100_000


def s2710(a, b, c, d, e, x, LEN_1D):
    n = int(LEN_1D)
    big = n > 10
    x0pos = x[0] > 0.0
    if n >= _SMALL:
        _s2710_par(a, b, c, d, e, n, big, x0pos)
    else:
        _s2710_serial(a, b, c, d, e, n, big, x0pos)
    return None


def _warmup():
    for n in (4096, 200_000):
        for big in (True, False):
            for x0pos in (True, False):
                aa = np.random.rand(n).astype(np.float64)
                bb = np.random.rand(n).astype(np.float64)
                cc = np.random.rand(n).astype(np.float64)
                dd = np.random.rand(n).astype(np.float64)
                ee = np.random.rand(n).astype(np.float64)
                _s2710_serial(aa, bb, cc, dd, ee, n, big, x0pos)
                aa[:] = np.random.rand(n)
                bb[:] = np.random.rand(n)
                _s2710_par(aa, bb, cc, dd, ee, n, big, x0pos)


import os as _os
_num_threads = int(__import__("os").environ.get("S2710_THREADS", "0"))
if _num_threads > 0:
    numba.set_num_threads(_num_threads)
_warmup()
