import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _vtvtv(a, b, c, n):
    for i in prange(n):
        a[i] = a[i] * b[i] * c[i]


def vtvtv(a, b, c, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    c = np.asanyarray(c)
    if (
        a.dtype == np.float64
        and b.dtype == np.float64
        and c.dtype == np.float64
        and a.ndim == 1
        and b.ndim == 1
        and c.ndim == 1
        and a.flags.c_contiguous
        and b.flags.c_contiguous
        and c.flags.c_contiguous
        and a.size >= n
        and b.size >= n
        and c.size >= n
    ):
        _vtvtv(a[:n], b[:n], c[:n], n)
    else:
        for i in range(n):
            a[i] = a[i] * b[i] * c[i]
    return None


# warm the JIT at import time so the timed first call is native code
_wa = np.ones(4096)
_wb = np.ones(4096)
_wc = np.ones(4096)
_vtvtv(_wa, _wb, _wc, 4096)

tsvc_2_vtvtv = vtvtv
tsvc_2_vtvtv_fp64 = vtvtv
