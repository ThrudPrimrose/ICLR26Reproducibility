import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _s2233(aa, bb, cc, n):
    # part A: column i scan along rows -- independent across i
    for i in prange(8, n):
        for j in range(8, n):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
    # part B: chain over rows, vectorized across columns
    for i in range(8, n):
        for j in range(8, n):
            bb[i, j] = bb[i - 1, j] + cc[i, j]

def s2233(aa, bb, cc, LEN_2D):
    _s2233(aa, bb, cc, int(LEN_2D))

# warm/compile at import time (not timed)
_a = np.zeros((16, 16)); _b = np.zeros((16, 16)); _c = np.zeros((16, 16))
_s2233(_a, _b, _c, 16)
