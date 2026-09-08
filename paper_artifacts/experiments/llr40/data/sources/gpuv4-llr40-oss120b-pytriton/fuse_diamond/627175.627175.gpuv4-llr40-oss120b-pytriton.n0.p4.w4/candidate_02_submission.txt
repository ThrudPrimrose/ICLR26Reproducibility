import numpy as np
import numba as nb

# Warm-up compile for float64 arrays (executed at import time).
# This ensures compilation overhead is not counted in the timed call.
@nb.njit(fastmath=True, parallel=True)
def _fuse_diamond_impl(out, a, LEN_1D):
    for i in nb.prange(LEN_1D):
        t = a[i] * a[i]
        out[i] = t * t - 1.0

# Trigger compilation for the typical dtype (float64).
_dummy_n = 1
_dummy_out = np.empty(_dummy_n, dtype=np.float64)
_dummy_a = np.empty(_dummy_n, dtype=np.float64)
_fuse_diamond_impl(_dummy_out, _dummy_a, _dummy_n)

def fuse_diamond(out, a, LEN_1D):
    """Compute out = a**4 - 1 using a JIT-compiled loop.

    Parameters
    ----------
    out : np.ndarray
        Output array, preallocated with shape (LEN_1D,).
    a : np.ndarray
        Input array, shape (LEN_1D,).
    LEN_1D : int
        Length of the arrays.
    Returns
    -------
    None
        Result is stored in ``out``.
    """
    # Ensure we operate on the correct slice in case the buffers are oversized.
    _fuse_diamond_impl(out[:LEN_1D], a[:LEN_1D], LEN_1D)
    return None
