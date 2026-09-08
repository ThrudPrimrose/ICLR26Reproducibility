import numpy as np
from numba import njit, prange

CHUNK = 2048


@njit(parallel=True, fastmath=True)
def _core(a, b, c, n):
    if n <= 0:
        return
    nb = (n + CHUNK - 1) // CHUNK
    for blk in prange(nb):
        i0 = blk * CHUNK
        m = CHUNK
        if i0 + m > n:
            m = n - i0
        s = np.empty(CHUNK)
        for j in range(m):
            s[j] = b[i0 + j] * c[i0 + j]
        if i0 == 0:
            a[i0] = s[0]
        else:
            a[i0] = s[0] + b[i0 - 1] * c[i0 - 1]
        for j in range(1, m):
            a[i0 + j] = s[j] + s[j - 1]


def s252(a, b, c, LEN_1D):
    _core(a, b, c, LEN_1D)


def _warm():
    n = 1 << 17
    x = np.arange(n, dtype=np.float64)
    _core(x, x, x, n)
    y = np.ones(3, dtype=np.float64)
    _core(y, y, y, 3)


_warm()
