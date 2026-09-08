"""Optimized s1244: in-place, numba parallel, two-pass with d as scratch.

Reference semantics (TSVC_2 s1244):
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]
        d[i] = a[i] + a[i+1]          # a[i+1] here is the ORIGINAL a
The read of a[i+1] is a forward WAR reference (value must be pre-loop).
Plan:
    pass 1 (parallel): d[i] = a[i+1]          (snapshot; d is overwritten anyway)
    pass 2 (parallel): a[i] = f(b[i],c[i]);  d[i] = f(...) + d[i]
No extra allocation, exactly 7 double-words of traffic per element.
"""
import numpy as np
import numba

_numba_njit = numba.njit(parallel=True, fastmath=False)


@_numba_njit
def _core(b, c, a, d, n):
    for i in numba.prange(n):
        d[i] = a[i + 1]
    for i in numba.prange(n):
        aa = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
        a[i] = aa
        d[i] = aa + d[i]


# Compile once at import time (off the timer).
def _warm():
    m = 64
    b = np.random.default_rng(0).random(m)
    c = np.random.default_rng(1).random(m)
    a = np.random.default_rng(2).random(m + 1)
    d = np.random.default_rng(3).random(m)
    _core(b, c, a, d, m)
    if not np.all(np.isfinite(a[:m])) or not np.all(np.isfinite(d)):
        raise RuntimeError("warmup check failed")


_warm()


def s1244(a, b, c, d, LEN_1D):
    n = LEN_1D - 1
    if n <= 0:
        return None
    if a.flags.c_contiguous and b.flags.c_contiguous and c.flags.c_contiguous \
            and d.flags.c_contiguous:
        _core(b, c, a, d, n)
    else:
        # Fallback for non-contiguous inputs.
        aa = b[:n] + c[:n] * c[:n] + b[:n] * b[:n] + c[:n]
        d[:n] = aa + a[1:LEN_1D]
        a[:n] = aa
    return None
