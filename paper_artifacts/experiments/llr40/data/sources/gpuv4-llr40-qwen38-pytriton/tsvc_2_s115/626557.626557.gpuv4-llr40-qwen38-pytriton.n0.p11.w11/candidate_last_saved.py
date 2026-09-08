import numpy as np
from numba import njit

@njit(cache=True, fastmath={'contract'})
def _k(a, aa, N):
    for j in range(N):
        aj = a[j]
        for i in range(j + 1, N):
            a[i] -= aa[j, i] * aj

def _warm():
    n = 16
    a = np.zeros(n, dtype=np.float64)
    aa = np.zeros((n, n), dtype=np.float64)
    _k(a, aa, n)
_warm()

def s115(a, aa, LEN_2D):
    _k(a, aa, LEN_2D)
    return None
