import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def fuse_stencil_through_transient(out, a, LEN_1D):
    """Fused stencil computation implemented with Numba for high performance.

    Computes out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
    for i in [1, LEN_1D-3]. The operation is performed in-place on ``out``.
    """
    # Parallel loop over the interior points.
    for i in prange(1, LEN_1D-2):
        out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
