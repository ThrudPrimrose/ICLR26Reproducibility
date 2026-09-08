import numpy as np
import numba

# Fused sequential Numba implementation of s2233.
# For each column, compute prefix sums for aa and bb simultaneously.

@numba.njit(fastmath=False)
def _fused_seq(aa, bb, cc, LEN_2D, offset):
    for i in range(offset, LEN_2D):
        acc_aa = aa[offset - 1, i]
        acc_bb = bb[offset - 1, i]
        for j in range(offset, LEN_2D):
            val = cc[j, i]
            acc_aa += val
            aa[j, i] = acc_aa
            acc_bb += val
            bb[j, i] = acc_bb

def s2233(aa, bb, cc, LEN_2D):
    """Fused Numba implementation of the s2233 kernel.

    Parameters
    ----------
    aa : np.ndarray
    bb : np.ndarray
    cc : np.ndarray
    LEN_2D : int
    """
    offset = 8
    if LEN_2D <= offset:
        return None
    _fused_seq(aa, bb, cc, LEN_2D, offset)
    return None
