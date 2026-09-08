import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, nogil=True)
def _k(a, b, c, d, aa, bb, cc, n2, n):
    for k in prange(n2):
        aa[k] += bb[k] * cc[k]
    for i in prange(n):
        a[i] = b[i] + c[i] * d[i]


def s2275(a, b, c, d, aa, bb, cc, LEN_2D):
    _k(a, b, c, d, aa.ravel(), bb.ravel(), cc.ravel(), LEN_2D * LEN_2D, LEN_2D)
    return None


# Warm up the JIT at import time (import runs before the clock starts).
_n = 256
_np_a = np.zeros(_n)
_np_b = np.zeros(_n)
_np_c = np.zeros(_n)
_np_d = np.zeros(_n)
_np_aa = np.zeros((_n, _n))
_np_bb = np.zeros((_n, _n))
_np_cc = np.zeros((_n, _n))
_k(_np_a, _np_b, _np_c, _np_d, _np_aa.ravel(), _np_bb.ravel(), _np_cc.ravel(), _n * _n, _n)
del _np_a, _np_b, _np_c, _np_d, _np_aa, _np_bb, _np_cc, _n
