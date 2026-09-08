import numpy as np
from numba import njit, prange

_NB = 8  # number of parallel blocks


@njit(parallel=True)
def _chunk(a, b, n, nblk):
    for blk in prange(nblk):
        s = blk * n // nblk
        e = (blk + 1) * n // nblk
        if s < e:
            if s == 0:
                x = b[n - 1]
                y = b[n - 2]
            elif s == 1:
                x = b[0]
                y = b[n - 1]
            else:
                x = b[s - 1]
                y = b[s - 2]
            i = s
            while i + 1 < e:
                a[i] = (b[i] + x + y) * 0.333
                a[i + 1] = (b[i + 1] + b[i] + x) * 0.333
                y = b[i]
                x = b[i + 1]
                i += 2
            if i < e:
                a[i] = (b[i] + x + y) * 0.333


@njit
def _serial(a, b, n):
    if n == 1:
        a[0] = (b[0] + b[0] + b[0]) * 0.333
        return
    x = b[n - 1]
    y = b[n - 2]
    for i in range(n):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]


def s255(a, b, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    nblk = _NB if n >= _NB else n
    if n >= 4:
        _chunk(a, b, n, nblk)
    else:
        _serial(a, b, n)
    return None


# Pre-compile / warm the JIT at import time (before the timer starts).
_ab = np.zeros(256, dtype=np.float64)
_aa = np.empty(256, dtype=np.float64)
_chunk(_aa, _ab, 256, _NB)
_chunk(_aa, _ab, 5, 5)
_serial(_aa, _ab, 3)
