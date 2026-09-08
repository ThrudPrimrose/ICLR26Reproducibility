"""TSVC tsvc_2 s1232 -- optimized python arm.

Reference (numpy):
    for j in range(LEN_2D):
        for i in range(j * VLEN, LEN_2D):
            aa[i, j] = bb[i, j] + cc[i, j]

The (j outer, i inner) iteration space is triangular; writing aa[i, j] means the
inner loop strides by LEN_2D * 8 bytes, so every load is a fresh cache line.
Interchanging the loops (i outer, j inner) makes the inner loop contiguous in
memory: for a given row i, the valid columns are exactly j = 0 .. i//VLEN.

Numba-compiled (parallel over rows) at import time so the JIT cost is never
inside the timed section.
"""
import numpy as np
import numba as nb
from numba import prange


@nb.njit(fastmath=True, parallel=True, cache=True)
def _s1232_kernel(aa, bb, cc, N, VLEN):
    for i in prange(N):
        lim = i // VLEN + 1
        arow = aa[i]
        brow = bb[i]
        crow = cc[i]
        for j in range(lim):
            arow[j] = brow[j] + crow[j]


def _warm():
    n = 48
    aa = np.zeros((n, n), dtype=np.float64)
    bb = np.ones((n, n), dtype=np.float64)
    cc = np.ones((n, n), dtype=np.float64)
    for vlen in (1, 3, 8):
        _s1232_kernel(aa, bb, cc, n, vlen)
        _s1232_kernel(aa, bb, cc, 0, vlen)
        _s1232_kernel(aa, bb, cc, 1, vlen)
        _s1232_kernel(aa, bb, cc, n, n * 7 + 1)


_warm()


def s1232(aa, bb, cc, LEN_2D, VLEN):
    _s1232_kernel(aa, bb, cc, LEN_2D, VLEN)
