"""Optimized s323 (TSVC_2 tsvc_2 s323):
    for i in 1..n-1:
        a[i] = b[i-1] + c[i]*d[i]
        b[i] = a[i]  + c[i]*e[i]
which reduces to the scan  b[i] = b[0] + sum_{j<=i} c[j]*(d[j]+e[j]),
a[i] = b[i] - c[i]*e[i].
Implemented as a two-phase block-parallel prefix sum (numba prange),
pre-compiled at import time so no JIT cost lands in the timed region.
"""
import os
import numpy as np
from numba import njit, prange

_NTHREADS = max(1, os.cpu_count() or 1)


@njit(parallel=True, fastmath=True)
def _scan_par(a, b, c, d, e, B):
    n = a.shape[0]
    m = n - 1
    nb = (m + B - 1) // B
    blks = np.empty(nb, np.float64)
    b0 = b[0]
    # phase 1: per-block local prefixes of t[i] = c[i]*(d[i]+e[i]) into b[i]
    for blk in prange(nb):
        s = 0.0
        st = blk * B + 1
        en = st + B
        if en > n:
            en = n
        for i in range(st, en):
            s += c[i] * (d[i] + e[i])
            b[i] = s
        blks[blk] = s
    # sequential scan of block totals
    tot = 0.0
    for blk in range(nb):
        s = blks[blk]
        blks[blk] = tot
        tot += s
    # phase 2: add offsets, write final b and derive a
    for blk in prange(nb):
        off = blks[blk] + b0
        st = blk * B + 1
        en = st + B
        if en > n:
            en = n
        for i in range(st, en):
            bi = b[i] + off
            b[i] = bi
            a[i] = bi - c[i] * e[i]


@njit(fastmath=True)
def _scan_ser(a, b, c, d, e):
    n = a.shape[0]
    for i in range(1, n):
        ai = b[i - 1] + c[i] * d[i]
        a[i] = ai
        b[i] = ai + c[i] * e[i]


def _ref_np(a, b, c, d, e):
    """Pure-numpy fallback (any dtype/stride), vectorized + cumsum."""
    n = a.shape[0]
    if n > 1:
        np.multiply(c[1:], d[1:] + e[1:], out=b[1:])
        np.cumsum(b[1:], out=b[1:])
        b[1:] += b[0]
        np.multiply(c[1:], d[1:], out=a[1:])
        a[1:] += b[:n - 1]


def s323(a, b, c, d, e, LEN_1D):
    n = a.shape[0]
    if n <= 1:
        return None
    if (a.dtype == np.float64 and b.dtype == np.float64 and c.dtype == np.float64
            and d.dtype == np.float64 and e.dtype == np.float64
            and a.flags["C_CONTIGUOUS"] and b.flags["C_CONTIGUOUS"]
            and c.flags["C_CONTIGUOUS"] and d.flags["C_CONTIGUOUS"]
            and e.flags["C_CONTIGUOUS"]):
        if n >= 262144:
            _scan_par(a, b, c, d, e, max(512, n // (10 * _NTHREADS)))
        else:
            _scan_ser(a, b, c, d, e)
        return None
    _ref_np(a, b, c, d, e)
    return None


def _warm():
    z = np.zeros(16, dtype=np.float64)
    a = z.copy()
    b = z.copy()
    _scan_ser(a, b, z, z, z)
    n = 1 << 22
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.ones(n)
    d = np.ones(n)
    e = np.ones(n)
    _scan_par(a, b, c, d, e, 4096)
    # a second call so the OMP pool is up and the dispatch is hot
    _scan_par(a, b, c, d, e, 4096)


_warm()
