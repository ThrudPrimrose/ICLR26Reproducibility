import numpy as np
import numba as nb

# Numba fused kernel: compute b = d*e and update a += b*c in a single pass.
@nb.njit(parallel=True, fastmath=True)
def _s152_fused(a, b, c, d, e, n):
    for i in nb.prange(n):
        prod = d[i] * e[i]
        b[i] = prod
        a[i] = a[i] + prod * c[i]

# Warm-up the JIT compilation once at import time (outside timed region).
_warmup_done = False
def _warmup():
    global _warmup_done
    if not _warmup_done:
        # small dummy arrays for compilation
        n = 8
        a = np.zeros(n, dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        c = np.ones(n, dtype=np.float64)
        d = np.arange(n, dtype=np.float64) + 1.0
        e = np.arange(n, dtype=np.float64) + 2.0
        _s152_fused(a, b, c, d, e, n)
        _warmup_done = True

_warmup()

def s152(a, b, c, d, e, LEN_1D):
    """In-place kernel for TSVC tsvc_2 s152.
    Updates arrays `a` and `b`. Parameters are NumPy arrays of length LEN_1D.
    """
    _s152_fused(a, b, c, d, e, LEN_1D)
    return None
