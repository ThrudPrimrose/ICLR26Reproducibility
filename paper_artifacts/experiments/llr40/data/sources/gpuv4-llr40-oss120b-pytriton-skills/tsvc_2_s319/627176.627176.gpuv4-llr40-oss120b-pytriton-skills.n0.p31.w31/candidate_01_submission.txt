import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _s319_impl(a, b, c, d, e, LEN_1D):
    sum_val = 0.0
    for i in prange(LEN_1D):
        ai = c[i] + d[i]
        bi = c[i] + e[i]
        a[i] = ai
        b[i] = bi
        sum_val += ai + bi
    b[0] = sum_val

def _warmup():
    n = 1
    a = np.empty(n, dtype=np.float64)
    b = np.empty(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    d = np.empty(n, dtype=np.float64)
    e = np.empty(n, dtype=np.float64)
    _s319_impl(a, b, c, d, e, n)

_warmup()

def s319(a, b, c, d, e, LEN_1D):
    _s319_impl(a, b, c, d, e, LEN_1D)
    return None
