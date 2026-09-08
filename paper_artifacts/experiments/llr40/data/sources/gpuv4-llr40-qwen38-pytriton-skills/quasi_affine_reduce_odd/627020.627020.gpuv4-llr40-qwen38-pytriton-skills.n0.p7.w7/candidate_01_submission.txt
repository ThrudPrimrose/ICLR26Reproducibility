import os
import numpy as np

os.environ["NUMBA_NUM_THREADS"] = "512"
import numba  # noqa: E402
from numba import njit, prange  # noqa: E402

@njit(parallel=True)
def _sodd(a, n):
    acc = 0.0
    n2 = n // 2
    for j in prange(n2):
        acc += a[2 * j + 1]
    return acc

_dummy = np.zeros(1024)
_sodd(_dummy, 1024)
del _dummy

def quasi_affine_reduce_odd(a, out, LEN_1D):
    out[0] = _sodd(a, LEN_1D)
    return None
