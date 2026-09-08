import numpy as np
import numba
from numba import prange

def fuse_move_ifs(a, b, src, cond, LEN_2D, K):
    """Optimized implementation of fuse_move_ifs using Numba parallel loops.
    
    Parameters
    ----------
    a : ndarray, shape (LEN_2D, LEN_2D)
        Output array written where `cond[i] > 0`.
    b : ndarray, shape (LEN_2D, LEN_2D)
        Output array written when `K > 0`.
    src : ndarray, shape (LEN_2D, LEN_2D)
        Input source array.
    cond : ndarray, shape (LEN_2D,)
        Condition per row.
    LEN_2D : int
        Size of the 2D dimensions (redundant, shape of arrays).
    K : int
        Flag controlling the second write.
    """
    _fused_impl(a, b, src, cond, LEN_2D, K)
    return None

# Numba JIT compiled fused loop. Uses parallel execution over rows.
@numba.njit(parallel=True, fastmath=True)
def _fused_impl(a, b, src, cond, LEN_2D, K):
    for i in prange(LEN_2D):
        flag = cond[i] > 0.0
        for j in range(LEN_2D):
            val = src[i, j]
            if flag:
                a[i, j] = val * 2.0
            if K > 0:
                b[i, j] = val + 1.0
