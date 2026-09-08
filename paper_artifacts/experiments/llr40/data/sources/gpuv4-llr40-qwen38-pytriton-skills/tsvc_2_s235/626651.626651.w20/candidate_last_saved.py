"""Optimized TSVC tsvc_2/s235: elementwise a update + column-wise scan.

    a[i] += b[i]*c[i]
    for j = 1..N-1: aa[j,i] = aa[j-1,i] + bb[j,i]*a[i]

The j-loop is a per-column prefix recurrence; columns are independent, so we
parallelize over groups of 8 consecutive columns.  8 doubles = one 64-byte
cache line, so each tile reads/writes whole lines (no stride amplification),
and the 8 recurrence chains give enough ILP to stay memory-bound.
"""
import numpy as np
import numba as nb


@nb.njit(parallel=True, fastmath=False, nogil=True)
def _core(a, b, c, aa, bb, N):
    full = (N >> 3) << 3
    ntile = full >> 3
    for t in nb.prange(ntile):
        i0 = t << 3
        i1 = i0 + 1
        i2 = i0 + 2
        i3 = i0 + 3
        i4 = i0 + 4
        i5 = i0 + 5
        i6 = i0 + 6
        i7 = i0 + 7
        ai0 = a[i0] + b[i0] * c[i0]; a[i0] = ai0
        ai1 = a[i1] + b[i1] * c[i1]; a[i1] = ai1
        ai2 = a[i2] + b[i2] * c[i2]; a[i2] = ai2
        ai3 = a[i3] + b[i3] * c[i3]; a[i3] = ai3
        ai4 = a[i4] + b[i4] * c[i4]; a[i4] = ai4
        ai5 = a[i5] + b[i5] * c[i5]; a[i5] = ai5
        ai6 = a[i6] + b[i6] * c[i6]; a[i6] = ai6
        ai7 = a[i7] + b[i7] * c[i7]; a[i7] = ai7
        acc0 = aa[0, i0]
        acc1 = aa[0, i1]
        acc2 = aa[0, i2]
        acc3 = aa[0, i3]
        acc4 = aa[0, i4]
        acc5 = aa[0, i5]
        acc6 = aa[0, i6]
        acc7 = aa[0, i7]
        for j in range(1, N):
            acc0 += bb[j, i0] * ai0
            acc1 += bb[j, i1] * ai1
            acc2 += bb[j, i2] * ai2
            acc3 += bb[j, i3] * ai3
            acc4 += bb[j, i4] * ai4
            acc5 += bb[j, i5] * ai5
            acc6 += bb[j, i6] * ai6
            acc7 += bb[j, i7] * ai7
            aa[j, i0] = acc0
            aa[j, i1] = acc1
            aa[j, i2] = acc2
            aa[j, i3] = acc3
            aa[j, i4] = acc4
            aa[j, i5] = acc5
            aa[j, i6] = acc6
            aa[j, i7] = acc7
    for i in nb.prange(full, N):
        ai = a[i] + b[i] * c[i]
        a[i] = ai
        acc = aa[0, i]
        for j in range(1, N):
            acc += bb[j, i] * ai
            aa[j, i] = acc


def _warm():
    for N in (16, 13, 1, 0):
        a = np.ones(N)
        b = np.ones(N)
        c = np.ones(N)
        aa = np.ones((N, N))
        bb = np.ones((N, N))
        _core(a, b, c, aa, bb, N)


try:
    _warm()
except Exception:
    pass


def s235(a, b, c, aa, bb, LEN_2D):
    N = int(LEN_2D)
    if not (a.flags.c_contiguous and b.flags.c_contiguous and c.flags.c_contiguous
            and aa.flags.c_contiguous and bb.flags.c_contiguous):
        a0 = a.copy()
        b0 = b.copy()
        c0 = c.copy()
        aa0 = aa.copy()
        bb0 = bb.copy()
        _core(a0, b0, c0, aa0, bb0, N)
        a[...] = a0
        aa[...] = aa0
        return None
    _core(a, b, c, aa, bb, N)
    return None
