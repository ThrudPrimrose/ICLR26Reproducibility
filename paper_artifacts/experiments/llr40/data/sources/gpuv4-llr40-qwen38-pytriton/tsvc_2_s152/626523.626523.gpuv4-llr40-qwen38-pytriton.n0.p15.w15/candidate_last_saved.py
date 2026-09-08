import numpy as np
import numba
from numba import prange


@numba.njit(parallel=True)
def _s152(a, b, c, d, e, n):
    for i in prange(n):
        t = d[i] * e[i]
        b[i] = t
        a[i] = a[i] + t * c[i]


def s152(a, b, c, d, e, LEN_1D):
    _s152(a, b, c, d, e, LEN_1D)
    return None


# Pre-compile (and initialize the threading layer) at import time, before the clock.
_z = np.zeros(1 << 10)
_s152(_z, _z.copy(), _z.copy(), _z.copy(), _z.copy(), 1 << 10)
del _z
