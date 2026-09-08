import numpy as np
from numba import njit

@njit(fastmath=True)
def scan_affine_decay(y, c, x, LEN_1D):
    """In-place scan of first-order linear recurrence y[i] = c[i] * y[i-1] + x[i]."""
    if LEN_1D == 0:
        return
    y[0] = x[0]
    for i in range(1, LEN_1D):
        y[i] = c[i] * y[i - 1] + x[i]
