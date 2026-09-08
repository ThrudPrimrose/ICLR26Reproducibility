import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=False)
def _s275_par(aa, bb, cc, n):
    for i in prange(n):
        if aa[0, i] > 0.0:
            acc = aa[0, i]
            for j in range(1, n):
                acc = acc + bb[j, i] * cc[j, i]
                aa[j, i] = acc


def s275(aa, bb, cc, LEN_2D):
    _s275_par(aa, bb, cc, LEN_2D)
    return None


# ---- import-time warmup: compile + thread-pool startup (not timed) ----
def _warm():
    n = 64
    a = np.zeros((n, n)); b = np.zeros((n, n)); c = np.zeros((n, n))
    a[0, :] = 1.0
    for _ in range(3):
        _s275_par(a, b, c, n)
    a[:2, 1::2] = 0.0  # exercise the disabled-column branch
    _s275_par(a, b, c, n)

_warm()
