"""TSVC s2710 -- optimized Python (numba) implementation.

Fused, branchy elementwise kernel: per-element work is independent, so the
loop is parallelized with prange across all available cores.  The module
pre-compiles both the serial and parallel variants at import time so no
JIT cost lands inside the timed region.

In-place ABI (matches the NumPy reference): a, b, c are mutated, None is
returned.  d, e, x are read-only inputs.
"""
import numpy as np
import numba
from numba import prange


@numba.njit
def _s2710_ser(a, b, c, d, e, n, big, pos):
    for i in range(n):
        ai = a[i]
        bi = b[i]
        if ai > bi:
            a[i] = ai + bi * d[i]
            if big:
                c[i] = c[i] + d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            ei = e[i]
            b[i] = ai + ei * ei
            if pos:
                c[i] = ai + d[i] * d[i]
            else:
                c[i] = c[i] + ei * ei


@numba.njit(parallel=True)
def _s2710_par(a, b, c, d, e, n, big, pos):
    for i in prange(n):
        ai = a[i]
        bi = b[i]
        if ai > bi:
            a[i] = ai + bi * d[i]
            if big:
                c[i] = c[i] + d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            ei = e[i]
            b[i] = ai + ei * ei
            if pos:
                c[i] = ai + d[i] * d[i]
            else:
                c[i] = c[i] + ei * ei


_PAR_MIN = 1 << 16


def s2710(a, b, c, d, e, x, LEN_1D):
    n = int(LEN_1D)
    big = n > 10
    pos = bool(x[0] > 0.0)
    if n >= _PAR_MIN:
        _s2710_par(a, b, c, d, e, n, big, pos)
    else:
        _s2710_ser(a, b, c, d, e, n, big, pos)
    return None


# Aliases in case the harness looks the symbol up under another name.
tsvc_2_s2710 = s2710
tsvc_2_s2710_fp64 = s2710


def _warm():
    n = 4096
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.ones(n)
    e = np.ones(n)
    x = np.array([1.0])
    _s2710_ser(a, b, c, d, e, n, True, True)
    _s2710_ser(a, b, c, d, e, n, False, False)
    _s2710_par(a, b, c, d, e, n, True, True)
    _s2710_par(a, b, c, d, e, n, False, False)


_warm()
