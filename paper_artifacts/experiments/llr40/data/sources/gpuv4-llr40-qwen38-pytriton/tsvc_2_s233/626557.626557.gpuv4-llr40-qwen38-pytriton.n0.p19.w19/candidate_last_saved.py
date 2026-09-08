"""TSVC tsvc_2 s233 -- optimized Python submission (v2: ILP unrolled scans).

Semantics (reference):
    for i in 8..n-1:
        for j in 8..n-1:  aa[j,i] = aa[j-1,i] + cc[j,i]   (scan down column i of aa)
        for j in 8..n-1:  bb[j,i] = bb[j,i-1] + cc[j,i]   (scan across row j of bb)

The two updates are independent; each is a set of independent prefix scans
(one per column for aa, one per row for bb).  Numba runs one parallel region
per family; each task carries FOUR independent scan chains at once so the
load -> add dependency latency is hidden by instruction-level parallelism.
The per-chain addition order is the reference's, so results are bit-exact.
"""
import numpy as np
import numba as nb


@nb.njit(parallel=True, boundscheck=False, fastmath=False)
def _aa4(aa, cc, n):
    # independent scans down 4 columns of aa at a time, seeded by row 7
    nblk = (n - 8) // 4
    for b in nb.prange(nblk):
        i0 = 8 + 4 * b
        v0 = aa[7, i0]
        v1 = aa[7, i0 + 1]
        v2 = aa[7, i0 + 2]
        v3 = aa[7, i0 + 3]
        for j in range(8, n):
            v0 += cc[j, i0]
            v1 += cc[j, i0 + 1]
            v2 += cc[j, i0 + 2]
            v3 += cc[j, i0 + 3]
            aa[j, i0] = v0
            aa[j, i0 + 1] = v1
            aa[j, i0 + 2] = v2
            aa[j, i0 + 3] = v3
    # tail columns (n - 8) % 4
    for i in range(8 + 4 * nblk, n):
        v = aa[7, i]
        for j in range(8, n):
            v += cc[j, i]
            aa[j, i] = v


@nb.njit(parallel=True, boundscheck=False, fastmath=False)
def _bb4(bb, cc, n):
    # independent scans across 4 rows of bb at a time, seeded by column 7
    nblk = (n - 8) // 4
    for b in nb.prange(nblk):
        j0 = 8 + 4 * b
        v0 = bb[j0, 7]
        v1 = bb[j0 + 1, 7]
        v2 = bb[j0 + 2, 7]
        v3 = bb[j0 + 3, 7]
        for i in range(8, n):
            v0 += cc[j0, i]
            v1 += cc[j0 + 1, i]
            v2 += cc[j0 + 2, i]
            v3 += cc[j0 + 3, i]
            bb[j0, i] = v0
            bb[j0 + 1, i] = v1
            bb[j0 + 2, i] = v2
            bb[j0 + 3, i] = v3
    for j in range(8 + 4 * nblk, n):
        v = bb[j, 7]
        for i in range(8, n):
            v += cc[j, i]
            bb[j, i] = v


# warm the JIT + thread pool at import time (not charged to the timed call)
_w = np.zeros((33, 33))
_aa4(_w, _w, 33)
_bb4(_w, _w, 33)


def s233(aa, bb, cc, LEN_2D):
    n = int(LEN_2D)
    if n <= 8:
        return None
    if aa.flags.c_contiguous and bb.flags.c_contiguous and cc.flags.c_contiguous:
        _aa4(aa, cc, n)
        _bb4(bb, cc, n)
        return None
    aac = np.ascontiguousarray(aa)
    bbc = np.ascontiguousarray(bb)
    ccc = np.ascontiguousarray(cc)
    _aa4(aac, ccc, n)
    _bb4(bbc, ccc, n)
    aa[...] = aac
    bb[...] = bbc
    return None
