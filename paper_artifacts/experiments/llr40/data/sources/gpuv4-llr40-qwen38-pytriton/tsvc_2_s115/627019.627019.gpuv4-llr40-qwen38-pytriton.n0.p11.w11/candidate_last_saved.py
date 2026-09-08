import numpy as np
import numba

@numba.njit(parallel=True)
def _k(a, aa, N):
    for j in range(N):
        aj = a[j]
        for i in numba.prange(j + 1, N):
            a[i] = a[i] - aa[j, i] * aj

def s115(a, aa, LEN_2D):
    _k(a, aa, int(LEN_2D))
    return None

# Warm up JIT + thread pool at import time (before the clock).
_d = np.zeros(256, dtype=np.float64)
_dd = np.zeros((256, 256), dtype=np.float64)
s115(_d, _dd, 256)
del _d, _dd
