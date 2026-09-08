"""Optimized python arm for tsvc_2_s235 (TSVC_2 s235, fp64).

Reference semantics (in-place on a and aa):
    for i in range(N):
        a[i] = a[i] + b[i] * c[i]
        for j in range(1, N):
            aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]

aa[:, i] is a prefix scan of bb[:, i] * a[i] seeded with aa[0, i], so columns
are independent: we parallelize over columns (prange) and access memory
contiguously through transposed views. FP order is kept exactly as in the
reference (one mul + one add per step, left to right), so results are
bit-identical to the numpy reference.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _core(a, b, c, aa, bb, N):
    if N < 1:
        return
    for i in prange(N):
        a[i] += b[i] * c[i]
    if N < 2:
        return
    at = aa.T
    bt = bb.T
    for i in prange(N):
        ai = a[i]
        r = at[i, 0]
        for j in range(1, N):
            r += bt[i, j] * ai
            at[i, j] = r


def s235(a, b, c, aa, bb, LEN_2D):
    N = int(LEN_2D)
    A = np.ascontiguousarray(a, dtype=np.float64)
    B = np.ascontiguousarray(b, dtype=np.float64)
    C = np.ascontiguousarray(c, dtype=np.float64)
    AA = np.ascontiguousarray(aa, dtype=np.float64)
    BB = np.ascontiguousarray(bb, dtype=np.float64)
    _core(A, B, C, AA, BB, N)
    if A is not a:
        a[...] = A
    if AA is not aa:
        aa[...] = AA
    return None


# Warm up (JIT compile + thread pool) at import time, before timing starts.
_w = 8
_wa = np.zeros(_w)
_wb = np.ones(_w)
_wc = np.ones(_w)
_waa = np.zeros((_w, _w))
_wbb = np.ones((_w, _w))
_core(_wa, _wb, _wc, _waa, _wbb, _w)
