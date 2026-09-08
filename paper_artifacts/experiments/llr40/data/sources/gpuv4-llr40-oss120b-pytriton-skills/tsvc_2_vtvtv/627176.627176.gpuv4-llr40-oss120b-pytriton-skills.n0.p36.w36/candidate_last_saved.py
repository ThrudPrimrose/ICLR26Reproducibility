# Simple Numba JIT implementation matching baseline
import numpy as np
import numba

@numba.njit(fastmath=True)
def _vtvtv(a, b, c, n):
    for i in range(n):
        a[i] = a[i] * b[i] * c[i]

def vtvtv(a, b, c, LEN_1D):
    _vtvtv(a, b, c, LEN_1D)
    return None
