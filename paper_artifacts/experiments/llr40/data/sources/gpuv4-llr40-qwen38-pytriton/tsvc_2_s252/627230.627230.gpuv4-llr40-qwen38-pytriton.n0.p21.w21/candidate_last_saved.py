"""TSVC tsvc_2 kernel ``s252`` optimized.

The reference loop is

    t = 0
    for i: s = b[i]*c[i]; a[i] = s + t; t = s

i.e. a[i] = b[i]*c[i] + b[i-1]*c[i-1]  (a[0] = b[0]*c[0]).  The loop-carried
value is just the previous *product*, so the loop is an embarrassingly
parallel shifted pair-op: no scan, no scratch buffer, minimum 24 B/element
of traffic (read b, read c, write a).

The kernel is compiled and the parallel pool is warmed at import time so the
timed call is pure compute.  Per-element math is two fp64 muls and one fp64
add with exactly the reference's rounding (no FMA contraction), so output is
bitwise identical to the C oracle.
"""
import numpy as np

try:
    from numba import njit, prange

    @njit(parallel=True, fastmath=False)
    def _k(a, b, c):
        n = a.shape[0]
        a[0] = b[0] * c[0]
        for i in prange(1, n):
            p = b[i] * c[i]
            q = b[i - 1] * c[i - 1]
            a[i] = p + q

    # compile + warm the thread pool before the clock starts
    _d = np.ones(1 << 20)
    _e = np.ones(1 << 20)
    _f = np.ones(1 << 20)
    _k(_d, _e, _f)
    _HAVE = True
except Exception:
    _HAVE = False


def s252(a, b, c, LEN_1D):
    if _HAVE:
        _k(a, b, c)
        return a
    sc = b * c
    a[0] = sc[0]
    if LEN_1D > 1:
        a[1:] = sc[1:] + sc[:-1]
    return a


tsvc_2_s252 = s252
