import numpy as np
import numba as nb

# Threshold for choosing parallel implementation (heuristic based on performance).
_PARALLEL_THRESHOLD = 2000000  # number of elements

# Non-parallel Numba implementation for small arrays.
@nb.njit(fastmath=True)
def _vpvts_nopar(a, b, S):
    for i in range(a.shape[0]):
        a[i] = a[i] + b[i] * S

# Parallel Numba implementation for large arrays.
@nb.njit(parallel=True, fastmath=True)
def _vpvts_par(a, b, S):
    for i in nb.prange(a.shape[0]):
        a[i] = a[i] + b[i] * S

# Warm up JIT compilation with a dummy call (size does not affect compiled signature).
_dummy_a = np.empty(1, dtype=np.float64)
_dummy_b = np.empty(1, dtype=np.float64)
_vpvts_nopar(_dummy_a, _dummy_b, 0.0)
_vpvts_par(_dummy_a, _dummy_b, 0.0)

def vpvts(a, b, LEN_1D, S):
    """In-place update of array ``a``: a[i] = a[i] + b[i] * S.

    ``LEN_1D`` is ignored; ``a`` and ``b`` are assumed to be length ``LEN_1D``.
    ``S`` may be any scalar broadcastable to the operation.
    The function modifies ``a`` in place and returns ``None``.
    """
    # Choose implementation based on array size to get the best performance.
    if a.shape[0] > _PARALLEL_THRESHOLD:
        _vpvts_par(a, b, S)
    else:
        _vpvts_nopar(a, b, S)
    return None
