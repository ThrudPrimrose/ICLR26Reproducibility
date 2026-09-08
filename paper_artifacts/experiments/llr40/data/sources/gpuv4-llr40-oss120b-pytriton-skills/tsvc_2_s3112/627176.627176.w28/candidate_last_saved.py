import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def _prefix_sum(a, b):
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i]
        b[i] = s

# Warm-up JIT compilation (once at import)
_dummy_a = np.empty(1, dtype=np.float64)
_dummy_b = np.empty(1, dtype=np.float64)
_prefix_sum(_dummy_a, _dummy_b)

def s3112(a, b, LEN_1D):
    """Compute prefix sum of a into b (in-place).
    Parameters
    ----------
    a : np.ndarray
        Input array of shape (LEN_1D,).
    b : np.ndarray
        Output array of shape (LEN_1D,). Will be overwritten with the prefix sum.
    LEN_1D : int
        Length of the arrays (redundant, but kept for API compatibility).
    """
    _prefix_sum(a, b)
    return None
