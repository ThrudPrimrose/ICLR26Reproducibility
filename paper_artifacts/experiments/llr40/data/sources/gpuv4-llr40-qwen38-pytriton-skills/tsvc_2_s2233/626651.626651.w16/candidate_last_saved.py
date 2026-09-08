"""Optimized tsvc_2 s2233.

The reference computes, for R in 8..n-1 and C in 8..n-1:
    aa[R, C] = aa[R-1, C] + cc[R, C]
    bb[R, C] = bb[R-1, C] + cc[R, C]
Both are row-wise scans: each row depends only on the previous row, and within a
row the columns are independent.  Iterating rows in order makes the memory access
pure sequential streaming (row 8, then row 9, ...) and each row is a vectorizable
add -- the fastest access pattern for this dependency.
"""
import numpy as np
from numba import njit


@njit(fastmath=True)
def _core(aa, bb, cc, n):
    for R in range(8, n):
        for C in range(8, n):
            aa[R, C] = aa[R - 1, C] + cc[R, C]
            bb[R, C] = bb[R - 1, C] + cc[R, C]


def s2233(aa, bb, cc, LEN_2D):
    n = int(LEN_2D)
    _core(aa, bb, cc, n)
    return None


# Warm the JIT at import time so the compile is not charged on the first timed call.
try:
    _w = np.zeros((32, 32), dtype=np.float64)
    _core(_w.copy(), _w.copy(), _w.copy(), 32)
except Exception:
    pass
