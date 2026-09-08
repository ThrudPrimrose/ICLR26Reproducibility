import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True)
def _col_pass(aa, cc, N):
    # aa[j,i] = aa[j-1,i] + cc[j,i] : independent prefix sums down each column.
    # 8 columns per thread: one 64B cache line per j-step, 8 independent add chains.
    nb = (N - 8) // 8
    for t in prange(nb):
        i0 = 8 + 8 * t
        x0 = aa[7, i0]
        x1 = aa[7, i0 + 1]
        x2 = aa[7, i0 + 2]
        x3 = aa[7, i0 + 3]
        x4 = aa[7, i0 + 4]
        x5 = aa[7, i0 + 5]
        x6 = aa[7, i0 + 6]
        x7 = aa[7, i0 + 7]
        for j in range(8, N):
            x0 += cc[j, i0]
            x1 += cc[j, i0 + 1]
            x2 += cc[j, i0 + 2]
            x3 += cc[j, i0 + 3]
            x4 += cc[j, i0 + 4]
            x5 += cc[j, i0 + 5]
            x6 += cc[j, i0 + 6]
            x7 += cc[j, i0 + 7]
            aa[j, i0] = x0
            aa[j, i0 + 1] = x1
            aa[j, i0 + 2] = x2
            aa[j, i0 + 3] = x3
            aa[j, i0 + 4] = x4
            aa[j, i0 + 5] = x5
            aa[j, i0 + 6] = x6
            aa[j, i0 + 7] = x7
    for i0 in range(8 + 8 * nb, N):
        x = aa[7, i0]
        for j in range(8, N):
            x += cc[j, i0]
            aa[j, i0] = x


@njit(parallel=True, nogil=True)
def _row_pass(bb, cc, N):
    # bb[j,i] = bb[j,i-1] + cc[j,i] : independent prefix sums across each row.
    # 8 rows per thread: one 64B cache line per row per i-step, 8 independent chains.
    nb = (N - 8) // 8
    for t in prange(nb):
        j0 = 8 + 8 * t
        x0 = bb[j0, 7]
        x1 = bb[j0 + 1, 7]
        x2 = bb[j0 + 2, 7]
        x3 = bb[j0 + 3, 7]
        x4 = bb[j0 + 4, 7]
        x5 = bb[j0 + 5, 7]
        x6 = bb[j0 + 6, 7]
        x7 = bb[j0 + 7, 7]
        for i in range(8, N):
            x0 += cc[j0, i]
            x1 += cc[j0 + 1, i]
            x2 += cc[j0 + 2, i]
            x3 += cc[j0 + 3, i]
            x4 += cc[j0 + 4, i]
            x5 += cc[j0 + 5, i]
            x6 += cc[j0 + 6, i]
            x7 += cc[j0 + 7, i]
            bb[j0, i] = x0
            bb[j0 + 1, i] = x1
            bb[j0 + 2, i] = x2
            bb[j0 + 3, i] = x3
            bb[j0 + 4, i] = x4
            bb[j0 + 5, i] = x5
            bb[j0 + 6, i] = x6
            bb[j0 + 7, i] = x7
    for j0 in range(8 + 8 * nb, N):
        x = bb[j0, 7]
        for i in range(8, N):
            x += cc[j0, i]
            bb[j0, i] = x


def s233(aa, bb, cc, LEN_2D):
    n = int(LEN_2D)
    if n <= 8:
        return None
    _col_pass(aa, cc, n)
    _row_pass(bb, cc, n)
    return None


# Warm the JIT at import time (outside the timed section).
_d = np.zeros((16, 16))
s233(_d.copy(), _d.copy(), _d.copy(), 16)
del _d
