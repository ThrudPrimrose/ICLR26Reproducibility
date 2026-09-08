import numpy as np
import numba

# Numba-accelerated implementation of the scan_affine_decay recurrence.
# The function signature matches the reference: y, c, x are NumPy arrays of length LEN_1D,
# and LEN_1D is the problem size. The routine updates y in-place.

@numba.njit(fastmath=True, inline='always')
def _scan_affine_decay(y, c, x, LEN_1D):
    # Guard against empty arrays.
    if LEN_1D == 0:
        return
    # Seed the recurrence.
    y[0] = x[0]
    # Main loop – carries the data dependence.
    for i in range(1, LEN_1D):
        # y[i] = c[i] * y[i-1] + x[i]
        y[i] = c[i] * y[i - 1] + x[i]
    # No return value; y is modified in-place.

def scan_affine_decay(y, c, x, LEN_1D):
    """In-place scan with variable coefficient.
    Wrapper that forwards to the Numba-compiled implementation.
    """
    _scan_affine_decay(y, c, x, LEN_1D)
    return None
