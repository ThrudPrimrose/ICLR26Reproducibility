"""TSVC tsvc_2 kernel ``vtvtv`` -- optimized Python (numba) implementation.

Reference semantics (in-place on ``a``):
    for i in range(LEN_1D):
        a[i] = a[i] * b[i] * c[i]
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _vtvtv_par(a, b, c, n):
    for i in prange(n):
        a[i] = a[i] * b[i] * c[i]


@njit(fastmath=True)
def _vtvtv_ser(a, b, c, n):
    for i in range(n):
        a[i] = a[i] * b[i] * c[i]


def _warm():
    z = np.zeros(1024, dtype=np.float64)
    _vtvtv_par(z, z, z, 1024)
    _vtvtv_ser(z, z, z, 1024)
    z3 = np.zeros(3, dtype=np.float64)
    _vtvtv_par(z3, z3, z3, 3)
    _vtvtv_ser(z3, z3, z3, 3)
    _vtvtv_par(z, z, z, 0)
    _vtvtv_ser(z, z, z, 0)


_warm()


def vtvtv(a, b, c, LEN_1D):
    n = LEN_1D
    if (
        a.dtype is np.dtype(np.float64)
        and b.dtype is np.dtype(np.float64)
        and c.dtype is np.dtype(np.float64)
        and a.flags.c_contiguous
        and b.flags.c_contiguous
        and c.flags.c_contiguous
    ):
        if n >= 65536:
            _vtvtv_par(a, b, c, n)
        else:
            _vtvtv_ser(a, b, c, n)
        return None
    # Fallback: non-contiguous or unusual dtype -- reference-exact elementwise.
    a[...] = a * b * c
    return None
