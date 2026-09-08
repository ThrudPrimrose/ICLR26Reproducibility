import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=False)
def _scan(c, d, e, a, b, N, nb):
    M = N - 1
    los = np.empty(nb, dtype=np.int64)
    his = np.empty(nb, dtype=np.int64)
    for j in range(nb):
        los[j] = 1 + (j * M) // nb
        his[j] = 1 + ((j + 1) * M) // nb
    sums = np.empty(nb)
    for j in prange(nb):
        s = 0.0
        for k in range(los[j], his[j]):
            x = c[k] * d[k]
            y = c[k] * e[k]
            s = (s + x) + y
        sums[j] = s
    offs = np.empty(nb)
    off = b[0]
    for j in range(nb):
        offs[j] = off
        off = off + sums[j]
    for j in prange(nb):
        r = offs[j]
        for k in range(los[j], his[j]):
            x = c[k] * d[k]
            y = c[k] * e[k]
            a[k] = r + x
            r = a[k] + y
            b[k] = r


def s323(a, b, c, d, e, LEN_1D):
    N = int(LEN_1D)
    M = N - 1
    if M <= 0:
        return
    nb = min(16384, M)
    _scan(c, d, e, a, b, N, nb)


_w = np.ones(4096, dtype=np.float64)
s323(_w.copy(), _w.copy(), _w.copy(), _w.copy(), _w.copy(), 4096)
