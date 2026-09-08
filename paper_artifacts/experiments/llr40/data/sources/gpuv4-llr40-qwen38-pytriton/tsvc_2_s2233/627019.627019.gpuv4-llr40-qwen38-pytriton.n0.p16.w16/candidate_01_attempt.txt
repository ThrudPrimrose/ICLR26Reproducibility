import numpy as np
import numba
from numba import prange


@numba.njit(parallel=True, fastmath=False)
def _scan_fused(aa, bb, cc, N):
    for c in prange(8, N):
        acc_a = aa[7, c]
        acc_b = bb[7, c]
        for r in range(8, N):
            v = cc[r, c]
            acc_a += v
            acc_b += v
            aa[r, c] = acc_a
            bb[r, c] = acc_b


def _warm():
    n = 64
    aa = np.zeros((n, n))
    bb = np.zeros((n, n))
    cc = np.zeros((n, n))
    _scan_fused(aa, bb, cc, n)


_warm()


def s2233(aa, bb, cc, LEN_2D):
    n = int(LEN_2D)
    if n > 8:
        _scan_fused(aa, bb, cc, n)
    return None
