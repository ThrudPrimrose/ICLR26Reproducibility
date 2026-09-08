import numpy as np
from numba import njit, prange

_B = 4096
_scratch = {}


@njit(parallel=True, cache=False)
def _kernel(c, d, e, b, a, n, B, blocksum):
    N = n - 1
    if N <= 0:
        return
    nb = (N + B - 1) // B
    # Stage 1: per-block local prefix of x=c*(d+e), stored in b[i]; block total in blocksum[j]
    for j in prange(nb):
        lo = j * B
        hi = lo + B
        if hi > N:
            hi = N
        s = 0.0
        for p in range(lo, hi):
            i = p + 1
            s += c[i] * (d[i] + e[i])
            b[i] = s
        blocksum[j] = s
    # Stage 2: serial scan of block totals
    for j in range(1, nb):
        blocksum[j] += blocksum[j - 1]
    # Stage 3: add offset, finalize b and a
    b0 = b[0]
    for j in prange(nb):
        lo = j * B
        hi = lo + B
        if hi > N:
            hi = N
        off = blocksum[j - 1] if j > 0 else 0.0
        add = b0 + off
        for p in range(lo, hi):
            i = p + 1
            bi = b[i] + add
            b[i] = bi
            a[i] = bi - c[i] * e[i]


def s323(a, b, c, d, e, LEN_1D):
    n = int(LEN_1D)
    if n <= 1:
        return None
    N = n - 1
    nb = (N + _B - 1) // _B
    bs = _scratch.get(nb)
    if bs is None:
        bs = np.empty(nb)
        _scratch[nb] = bs
    _kernel(c, d, e, b, a, n, _B, bs)
    return None
