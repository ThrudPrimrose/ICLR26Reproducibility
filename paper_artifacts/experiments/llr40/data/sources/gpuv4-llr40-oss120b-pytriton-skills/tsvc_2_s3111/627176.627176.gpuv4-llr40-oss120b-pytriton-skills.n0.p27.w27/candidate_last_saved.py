"""Optimized Python implementation for the TSVC kernel ``s3111`` using Numba.

The kernel computes the sum of the positive entries of a one‑dimensional array ``a`` and
stores the result in ``b[0]``.  A pure NumPy implementation incurs two passes over the data –
first to build a boolean mask and then to sum – which is slower than a single‑pass reduction.

We instead use a Numba ``njit`` function with ``parallel=True`` which generates native code
that iterates over the array once and performs a parallel reduction.  The function is compiled
once at module import time (using a tiny dummy array) so the first timed call incurs no JIT
overhead.
"""

import numpy as np
import numba as nb

# Compile-time constant for a dummy warm‑up call (size 1). This runs at import time, outside the
# timed region, ensuring the heavy JIT compilation cost is not counted.
@nb.njit(parallel=True, fastmath=True)
def _sum_pos_parallel(a: np.ndarray) -> float:
    # Numba recognises ``s`` as a reduction variable and parallelises the loop.
    s = 0.0
    for i in nb.prange(a.shape[0]):
        val = a[i]
        if val > 0.0:
            s += val
    return s

# Warm up the JIT compilation for the given dtype and dimensionality.
_dummy = np.empty(1, dtype=np.float64)
_sum_pos_parallel(_dummy)  # noqa: F841 – the result is discarded.


def s3111(a: np.ndarray, b: np.ndarray, LEN_1D: int) -> None:
    """Sum positive elements of ``a`` into ``b[0]`` (in‑place ABI).

    Parameters
    ----------
    a : np.ndarray
        Input 1‑D array of ``float64`` values.
    b : np.ndarray
        Output buffer; ``b[0]`` receives the sum.
    LEN_1D : int
        Length of ``a`` – retained for API compatibility.
    """
    # The Numba implementation operates directly on the NumPy array without any Python‑level
    # loops, delivering a single‑pass, parallel reduction.
    b[0] = _sum_pos_parallel(a)

