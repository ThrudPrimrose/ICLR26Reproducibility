import numpy as np
import numba

@numba.njit(fastmath=True)
def s1244(a, b, c, d, LEN_1D):
    """In-place kernel for TSVC s1244.
    Computes a[i] = b[i] + c[i]^2 + b[i]^2 + c[i]
    and d[i] = a[i] + a[i+1] for i = 0 .. LEN_1D-2.
    The last elements of a and d are left unchanged.
    """
    if LEN_1D <= 1:
        return
    for i in range(LEN_1D - 1):
        a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
        d[i] = a[i] + a[i + 1]
    return None

# Warm up Numba compilation at import time to avoid JIT overhead during timed runs.
_dummy = np.empty(1, dtype=np.float64)
try:
    s1244(_dummy, _dummy, _dummy, _dummy, 1)
except Exception:
    pass
