import numpy as np
import numba

@numba.njit(fastmath=True, cache=True)
def _s252_impl(a, b, c):
    t = 0.0
    n = a.shape[0]
    for i in range(n):
        s = b[i] * c[i]
        a[i] = s + t
        t = s

# Trigger compilation at import time to populate the cache
_dummy = np.empty(1, dtype=np.float64)
_s252_impl(_dummy, _dummy, _dummy)

def s252(a, b, c, LEN_1D):
    """In-place kernel for TSVC s252 using a cached Numba JIT implementation."""
    _s252_impl(a, b, c)
    return None
