import numpy as np
import numba


@numba.njit(parallel=True)
def _s319(a, b, c, d, e, n):
    s = 0.0
    for i in numba.prange(n):
        ai = c[i] + d[i]
        a[i] = ai
        bi = c[i] + e[i]
        b[i] = bi
        s += ai + bi
    b[0] = s


def _f64(x):
    if x.dtype is np.dtype(np.float64) and x.flags.c_contiguous:
        return x, None
    return np.ascontiguousarray(x, dtype=np.float64), x


def s319(a, b, c, d, e, LEN_1D):
    n = int(LEN_1D)
    a2, a0 = _f64(a)
    b2, b0 = _f64(b)
    c2, _ = _f64(c)
    d2, _ = _f64(d)
    e2, _ = _f64(e)
    _s319(a2, b2, c2, d2, e2, n)
    if a0 is not None:
        a[...] = a2
    if b0 is not None:
        b[...] = b2
    return None


tsvc_2_s319 = s319
tsvc_2_s319_fp64 = s319

# Warm up: force the JIT compile now, before any timed call.
_w = np.ones(1024, dtype=np.float64)
_s319(_w.copy(), _w.copy(), _w.copy(), _w.copy(), _w.copy(), 1024)
