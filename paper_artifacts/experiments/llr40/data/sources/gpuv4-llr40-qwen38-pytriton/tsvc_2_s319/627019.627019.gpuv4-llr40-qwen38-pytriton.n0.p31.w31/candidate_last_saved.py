import numpy as np
from numba import njit, prange

# TSVC s319 (f64): a[i] = c[i]+d[i]; b[i] = c[i]+e[i]; b[0] = sum over i of (a[i]+b[i]).
# Single pass over data; prange over 8-element blocks (one AVX-512 double vector each),
# per-thread reduction, serial tail of < 8 elements.

@njit(fastmath=True, nogil=True, parallel=True)
def _s319_kernel(a, b, c, d, e):
    n = a.shape[0]
    nb = n // 8
    tot = 0.0
    for t in prange(nb):
        i0 = 8 * t
        v0 = c[i0]   + d[i0]
        v1 = c[i0+1] + d[i0+1]
        v2 = c[i0+2] + d[i0+2]
        v3 = c[i0+3] + d[i0+3]
        v4 = c[i0+4] + d[i0+4]
        v5 = c[i0+5] + d[i0+5]
        v6 = c[i0+6] + d[i0+6]
        v7 = c[i0+7] + d[i0+7]
        w0 = c[i0]   + e[i0]
        w1 = c[i0+1] + e[i0+1]
        w2 = c[i0+2] + e[i0+2]
        w3 = c[i0+3] + e[i0+3]
        w4 = c[i0+4] + e[i0+4]
        w5 = c[i0+5] + e[i0+5]
        w6 = c[i0+6] + e[i0+6]
        w7 = c[i0+7] + e[i0+7]
        a[i0]   = v0; a[i0+1] = v1; a[i0+2] = v2; a[i0+3] = v3
        a[i0+4] = v4; a[i0+5] = v5; a[i0+6] = v6; a[i0+7] = v7
        b[i0]   = w0; b[i0+1] = w1; b[i0+2] = w2; b[i0+3] = w3
        b[i0+4] = w4; b[i0+5] = w5; b[i0+6] = w6; b[i0+7] = w7
        tot += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 \
             + w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7
    s = 0.0
    i = 8 * nb
    while i < n:
        v = c[i] + d[i]
        w = c[i] + e[i]
        a[i] = v
        b[i] = w
        s += v
        s += w
        i += 1
    b[0] = tot + s


def _contig64(x):
    return (x.dtype == np.float64) and x.ndim == 1 and x.flags.c_contiguous


def s319(a, b, c, d, e, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    if not (_contig64(a) and _contig64(b) and _contig64(c)
            and _contig64(d) and _contig64(e)):
        ca = np.ascontiguousarray(a, dtype=np.float64)
        cb = np.ascontiguousarray(b, dtype=np.float64)
        cc = np.ascontiguousarray(c, dtype=np.float64)
        cd = np.ascontiguousarray(d, dtype=np.float64)
        ce = np.ascontiguousarray(e, dtype=np.float64)
        _s319_kernel(ca, cb, cc, cd, ce)
        a[...] = ca
        b[...] = cb
        return None
    _s319_kernel(a, b, c, d, e)
    return None


def _warm():
    n = 1 << 20
    a = np.empty(n, np.float64)
    b = np.empty(n, np.float64)
    c = np.random.random(n)
    d = np.random.random(n)
    e = np.random.random(n)
    s319(a, b, c, d, e, n)
    del a, b, c, d, e


_warm()
