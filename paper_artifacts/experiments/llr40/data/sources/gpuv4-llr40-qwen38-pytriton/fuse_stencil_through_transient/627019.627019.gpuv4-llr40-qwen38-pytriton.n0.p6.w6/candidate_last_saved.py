import numpy as np
import numba
from numba import njit, prange

# Unrolled-by-8 parallel kernel. Each block of 8 outputs out[i..i+7] reads
# a[i-1..i+9] (11 loads) and shares the 9 three-sums p0..p8. Every three-sum is
# evaluated left-to-right, bit-identical to the reference
#   out[i] = (a[i-1]+a[i]+a[i+1]) * (a[i]+a[i+1]+a[i+2])
# for i in 1..n-3.
@njit(parallel=True, fastmath=False, nogil=True)
def _kpar(a, out):
    n = a.shape[0]
    m = (n - 3) // 8
    for b in prange(m):
        i = 1 + 8 * b
        c0 = a[i - 1]
        c1 = a[i]
        c2 = a[i + 1]
        c3 = a[i + 2]
        c4 = a[i + 3]
        c5 = a[i + 4]
        c6 = a[i + 5]
        c7 = a[i + 6]
        c8 = a[i + 7]
        c9 = a[i + 8]
        c10 = a[i + 9]
        p0 = c0 + c1 + c2
        p1 = c1 + c2 + c3
        p2 = c2 + c3 + c4
        p3 = c3 + c4 + c5
        p4 = c4 + c5 + c6
        p5 = c5 + c6 + c7
        p6 = c6 + c7 + c8
        p7 = c7 + c8 + c9
        p8 = c8 + c9 + c10
        out[i] = p0 * p1
        out[i + 1] = p1 * p2
        out[i + 2] = p2 * p3
        out[i + 3] = p3 * p4
        out[i + 4] = p4 * p5
        out[i + 5] = p5 * p6
        out[i + 6] = p6 * p7
        out[i + 7] = p7 * p8
    for i in prange(1 + 8 * m, n - 2):
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2])


# Serial fallback for tiny inputs (avoids prange launch overhead).
@njit(fastmath=False, nogil=True)
def _kser(a, out):
    n = a.shape[0]
    for i in range(1, n - 2):
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2])


_SMALL = 1 << 14

# Pick the thread count: full available (affinity-clamped) pool up to 24.
try:
    numba.set_num_threads(min(24, numba.get_num_threads()))
except Exception:
    pass

# Precompile every signature used at call time so no timed rep ever compiles.
_w64 = np.zeros(16384 + 16)
_kpar(_w64.copy(), _w64.copy())
_kser(_w64.copy(), _w64.copy())
_w32 = np.zeros(16384 + 16, np.float32)
_kpar(_w32.copy(), _w32.copy())
_kser(_w32.copy(), _w32.copy())
del _w64, _w32


def _np_fallback(out, a, n):
    # Bit-identical vectorized reference (left-to-right adds per term).
    if n > 3:
        t1 = a[:-3] + a[1:-2]
        t1 = t1 + a[2:-1]
        t2 = a[1:-2] + a[2:-1]
        t2 = t2 + a[3:]
        out[1:-2] = t1 * t2


def fuse_stencil_through_transient(out, a, LEN_1D):
    n = int(LEN_1D)
    if (n >= _SMALL and a.ndim == 1 and out.ndim == 1
            and a.shape[0] == n and out.shape[0] == n
            and a.dtype == out.dtype
            and a.flags.c_contiguous and out.flags.c_contiguous):
        if n >= _SMALL * 8:
            _kpar(a, out)
        else:
            _kser(a, out)
        return None
    _np_fallback(out, a, n)
    return None
