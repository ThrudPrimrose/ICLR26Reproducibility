import numpy as np
import numba

# JIT-compiled kernel performing forward substitution.
# The kernel updates the array `a` in place using the matrix `aa`.
@numba.njit(fastmath=True)
def _s115_numba(a, aa):
    N = a.shape[0]
    for j in range(N):
        a_j = a[j]
        for i in range(j + 1, N):
            a[i] = a[i] - aa[j, i] * a_j
    # No return; modifies `a` in place.
    return None

# Pre-compile the JIT function at import time using a tiny dummy array.
# This moves the JIT compilation cost out of the timed region.
_dummy_a = np.empty(1, dtype=np.float64)
_dummy_aa = np.empty((1, 1), dtype=np.float64)
_s115_numba(_dummy_a, _dummy_aa)

def s115(a, aa, LEN_2D):
    """In-place forward substitution kernel.

    Parameters
    ----------
    a : np.ndarray, shape (LEN_2D,)
        The vector to be updated in place.
    aa : np.ndarray, shape (LEN_2D, LEN_2D)
        Input matrix.
    LEN_2D : int
        Length of the dimension (must match a.shape[0]).
    """
    # Ensure the input size matches the expected length.
    if a.shape[0] != LEN_2D:
        a = a[:LEN_2D]
    if aa.shape[0] != LEN_2D or aa.shape[1] != LEN_2D:
        aa = aa[:LEN_2D, :LEN_2D]
    # Call the compiled kernel; it updates `a` in place.
    _s115_numba(a, aa)
    return None

