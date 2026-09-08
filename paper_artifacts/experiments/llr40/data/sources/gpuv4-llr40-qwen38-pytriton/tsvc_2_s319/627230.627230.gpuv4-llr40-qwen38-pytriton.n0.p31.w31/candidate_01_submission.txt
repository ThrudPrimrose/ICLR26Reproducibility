from numba import njit, prange
import numpy as np

@njit(parallel=True)
def _run(a, b, c, d, e, LEN_1D):
    n = int(LEN_1D)
    s = 0.0
    for i in prange(n):
        a[i] = c[i] + d[i]
        b[i] = c[i] + e[i]
        s += a[i] + b[i]
    b[0] = s
    return None

def s319(a, b, c, d, e, LEN_1D):
    return _run(a, b, c, d, e, int(LEN_1D))

# warm-compile at import (untimed)
_w = np.empty(64, np.float64)
_wc = np.empty(64, np.float64)
_run(_w, _w.copy(), _wc, _wc, _wc, 64)
