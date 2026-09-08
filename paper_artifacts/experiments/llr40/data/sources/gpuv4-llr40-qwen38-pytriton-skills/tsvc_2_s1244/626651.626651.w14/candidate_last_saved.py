"""Optimized python arm for TSVC tsvc_2 s1244 (fp64, in-place ABI).

Reference semantics (loop i = 0..LEN_1D-2):
    a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]
    d[i] = a[i] + a[i+1]        # a[i+1] is the ORIGINAL value: it is
                                # only overwritten in the NEXT iteration

so:  a[i] = f(b[i], c[i])       for i < LEN_1D-1  (a[LEN_1D-1] untouched)
     d[i] = f(b[i], c[i]) + a_orig[i+1]

Both outputs are per-element functions of (b[i], c[i]) and a shift-read of
the input a; there is no true loop-carried dependence, so the work is fully
SIMD-vectorizable. numba emits AVX-512 (<8 x double>) loads/stores for the
loop. For large n the work is memory-bandwidth bound, so the big case runs
as a prange over independent elements (threads split the array); the small
case uses the serial kernel to avoid the parallel fork/join overhead.
"""

import numpy as np
import numba
from numba import prange

_THR = 65536  # below: serial; at/above: threaded


@numba.njit
def _s1244_ser(a, b, c, d, n):
    for i in range(n):
        an = a[i + 1]
        t = b[i] + c[i] * c[i]
        t = t + b[i] * b[i]
        t = t + c[i]
        a[i] = t
        d[i] = t + an


@numba.njit(parallel=True)
def _s1244_par(a, b, c, d, n):
    for i in prange(n):
        an = a[i + 1]
        t = b[i] + c[i] * c[i]
        t = t + b[i] * b[i]
        t = t + c[i]
        a[i] = t
        d[i] = t + an


def s1244(a, b, c, d, LEN_1D):
    n = LEN_1D - 1
    if n > 0:
        if n >= _THR:
            _s1244_par(a, b, c, d, n)
        else:
            _s1244_ser(a, b, c, d, n)
    return None


def _warm():
    # Pre-JIT both kernels (and both common dtypes) at import time so no
    # compilation ever falls inside the timed call.
    for dt in (np.float64, np.float32):
        m = 128
        a = np.arange(m, dtype=dt)
        b = np.arange(m, dtype=dt)
        c = np.arange(m, dtype=dt)
        d = np.zeros(m, dtype=dt)
        n = m - 1
        _s1244_ser(a, b, c, d, n)
        a2 = np.arange(2 * _THR, dtype=dt)
        b2 = a2.copy()
        c2 = a2.copy()
        d2 = np.zeros(2 * _THR, dtype=dt)
        _s1244_par(a2, b2, c2, d2, 2 * _THR - 1)


_warm()
