"""TSVC tsvc_2 s252 -- optimized.

Reference:
    t = 0.0
    for i in range(LEN_1D):
        s = b[i] * c[i]
        a[i] = s + t
        t = s

Unrolling the recurrence: a[i] = b[i]*c[i] + b[i-1]*c[i-1] (with the
b[-1]*c[-1] term taken as 0), so the loop has no loop-carried dependence and
is fully parallel.  Implemented as an 8-way unrolled numba kernel; the
JIT compile happens at import time (outside the timed region).
"""
import numpy as np

try:
    import numba

    @numba.njit(parallel=True, boundscheck=False, fastmath=False)
    def _s252(a, b, c, n):
        if n > 0:
            a[0] = b[0] * c[0]
        m = n >> 3
        for i in numba.prange(1, m):
            j = i << 3
            a[j + 0] = b[j + 0] * c[j + 0] + b[j - 1] * c[j - 1]
            a[j + 1] = b[j + 1] * c[j + 1] + b[j + 0] * c[j + 0]
            a[j + 2] = b[j + 2] * c[j + 2] + b[j + 1] * c[j + 1]
            a[j + 3] = b[j + 3] * c[j + 3] + b[j + 2] * c[j + 2]
            a[j + 4] = b[j + 4] * c[j + 4] + b[j + 3] * c[j + 3]
            a[j + 5] = b[j + 5] * c[j + 5] + b[j + 4] * c[j + 4]
            a[j + 6] = b[j + 6] * c[j + 6] + b[j + 5] * c[j + 5]
            a[j + 7] = b[j + 7] * c[j + 7] + b[j + 6] * c[j + 6]
        for i in range(1, min(8, n)):
            a[i] = b[i] * c[i] + b[i - 1] * c[i - 1]
        for i in range(n if n < 8 else m * 8, n):
            a[i] = b[i] * c[i] + b[i - 1] * c[i - 1]

    # Warm up the JIT at import time (before the clock starts).
    _z = np.zeros(64, dtype=np.float64)
    _s252(_z, _z, _z, 64)
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def s252(a, b, c, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    if _HAVE_NUMBA:
        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
            return a
        _s252(a, b, c, n)
        return None
    # numpy fallback
    s = b[:n] * c[:n]
    a[:n] = s
    if n > 1:
        a[1:n] += s[:n - 1]
    return None
