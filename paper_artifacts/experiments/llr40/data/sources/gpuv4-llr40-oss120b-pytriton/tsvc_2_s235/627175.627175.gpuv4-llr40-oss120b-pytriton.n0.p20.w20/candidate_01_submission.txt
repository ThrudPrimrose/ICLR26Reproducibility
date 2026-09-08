# Numba-accelerated implementation of TSVC kernel s235.
# Reference signature: s235(a, b, c, aa, bb, LEN_2D)
# - a, b, c: 1D float64 arrays of length LEN_2D
# - aa, bb: 2D float64 arrays of shape (LEN_2D, LEN_2D)
# - LEN_2D: integer dimension size
# This function updates a and aa in-place.

import numpy as np
import numba

# JIT-compiled implementation with parallel outer loop.
@numba.njit(parallel=True, fastmath=True)
def _s235_impl(a, b, c, aa, bb, LEN_2D):
    for i in numba.prange(LEN_2D):
        # Update a[i]
        a[i] = a[i] + b[i] * c[i]
        # Base value from aa[0, i]
        base = aa[0, i]
        # Accumulate contributions from bb rows
        acc = 0.0
        ai = a[i]  # updated a[i]
        for j in range(1, LEN_2D):
            acc += bb[j, i] * ai
            aa[j, i] = base + acc

def s235(a, b, c, aa, bb, LEN_2D):
    _s235_impl(a, b, c, aa, bb, LEN_2D)
    return None
