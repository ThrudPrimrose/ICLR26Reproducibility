import numpy as np
import numba
from numba import njit, prange


@njit(parallel=True, boundscheck=False, fastmath=True)
def _k(a, b, c, d, e, x, n):
    gt10 = n > 10
    xp = x[0] > 0.0
    for i in prange(n):
        if a[i] > b[i]:
            a[i] += b[i] * d[i]
            if gt10:
                c[i] += d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            if xp:
                c[i] = a[i] + d[i] * d[i]
            else:
                c[i] += e[i] * e[i]


def _warm():
    n = 2048
    a = np.zeros(n)
    b = np.ones(n)
    c = np.zeros(n)
    d = np.ones(n)
    e = np.zeros(n)
    _k(a, b, c, d, e, np.array([1.0]), n)   # n>10, x0>0
    _k(a, b, c, d, e, np.array([-1.0]), n)  # n>10, x0<0
    _k(a, b, c, d, e, np.array([1.0]), 8)   # n<=10, x0>0
    _k(a, b, c, d, e, np.array([-1.0]), 8)  # n<=10, x0<0
    return None


try:
    _warm()
except Exception:
    pass


def s2710(a, b, c, d, e, x, LEN_1D):
    _k(a, b, c, d, e, x, int(LEN_1D))
