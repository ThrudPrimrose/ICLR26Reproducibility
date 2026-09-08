# Optimized implementation of TSVC kernel s4112 using Numba JIT and parallel loops.
# This matches the reference signature:
#   def s4112(a, b, ip, LEN_1D):
# where a and b are 1-D float64 arrays, ip is a 1-D int32 (or int64) index array.
# The operation performed is: a[i] = a[i] + 2.0 * b[ip[i]] for i in range(LEN_1D).
# The function updates a in-place and returns None.

import numpy as np
import numba as nb

# Compile a fast, parallel implementation.
# fastmath enables LLVM fast-math optimizations.
@nb.njit(parallel=True, fastmath=True)
def _s4112_impl(a, b, ip, LEN_1D):
    for i in nb.prange(LEN_1D):
        a[i] = a[i] + b[ip[i]] * 2.0


def s4112(a, b, ip, LEN_1D):
    """In-place update of ``a`` according to the TSVC s4112 kernel.

    Parameters
    ----------
    a : np.ndarray (float64)
        Output array, length ``LEN_1D``. Modified in place.
    b : np.ndarray (float64)
        Input array, length ``LEN_1D``.
    ip : np.ndarray (int32 or int64)
        Index array, length ``LEN_1D``.
    LEN_1D : int
        Number of elements.
    """
    # The JIT-compiled implementation expects NumPy arrays of matching dtype.
    # No explicit checks are performed for performance reasons.
    _s4112_impl(a, b, ip, LEN_1D)
    return None

