import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def _sum_pos_serial(a, b, n):
    s = 0.0
    for i in range(n):
        if a[i] > 0.0:
            s += a[i]
    b[0] = s

@njit(parallel=True, fastmath=True)
def _sum_pos_par(a, b, n):
    s = 0.0
    for i in prange(n):
        if a[i] > 0.0:
            s += a[i]
    b[0] = s

def s3111(a, b, LEN_1D):
    n = int(LEN_1D)
    if n >= 2_000_000:
        _sum_pos_par(a, b, n)
    else:
        _sum_pos_serial(a, b, n)

tsvc_2_s3111_fp64 = s3111

# Warm up both paths at import time (not timed).
_wa = np.zeros(64)
_wb = np.zeros(2)
_sum_pos_serial(_wa, _wb, 64)
_sum_pos_par(_wa, _wb, 64)
