"""Optimized TSVC s2275 (python arm).

Reference semantics:
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            aa[j, i] = aa[j, i] + bb[j, i] * cc[j, i]
        a[i] = b[i] + c[i] * d[i]

i.e. a full-matrix FMA  aa += bb * cc  plus a vector FMA  a = b + c*d.

The parallel-numba baseline parallelizes the OUTER (i) loop, so for a fixed
i it walks aa[j, i] COLUMN-STRIDED: each step jumps a whole row (LEN_2D*8
bytes), touching one 8-byte element per 64-byte cache line (~8x DRAM
amplification).  The legality fix the puzzle points at is loop
distribution + interchange: move the matrix update out, make the inner walk
CONTIGUOUS.  Here the identical work is a single fused, multi-threaded
prange over flat row-major 1D views -- the minimum traffic (read A, B, C;
write A) in ONE streaming pass, which numba vectorizes to 8-wide AVX-512.

No thread count is touched: the process runs at numba's default (= the
number of available CPUs), which is optimal for this bandwidth-bound FMA and
keeps the baseline measured under identical conditions.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _flat(A, B, C, a, b, c, d, N):
    n2 = N * N
    for k in prange(n2):
        A[k] += B[k] * C[k]
    for i in prange(N):
        a[i] = b[i] + c[i] * d[i]


@njit(parallel=True, fastmath=True)
def _gen(a, b, c, d, aa, bb, cc, N):
    # Fallback for non-C-contiguous operands: interchange the loops so the
    # inner walk is row-contiguous, 2D indexing so writes land in the
    # caller's arrays.
    for j in prange(N):
        for i in range(N):
            aa[j, i] += bb[j, i] * cc[j, i]
    for i in prange(N):
        a[i] = b[i] + c[i] * d[i]


def s2275(a, b, c, d, aa, bb, cc, LEN_2D):
    N = int(LEN_2D)
    if aa.flags.c_contiguous and bb.flags.c_contiguous and cc.flags.c_contiguous:
        _flat(aa.ravel(), bb.ravel(), cc.ravel(), a, b, c, d, N)
    else:
        _gen(a, b, c, d, aa, bb, cc, N)


# Warm the JIT (and the threading pools) at import time, off the clock.
_w = np.ones(512)
_W = np.ones((512, 512))
_flat(_W.ravel(), _W.ravel(), _W.ravel(), _w, _w, _w, _w, 512)
_gen(_w, _w, _w, _w, _W, _W, _W, 512)
del _w, _W
