"""TSVC tsvc_2 kernel s119 (python optimization).

    aa[i, j] = aa[i - 1, j - 1] + bb[i, j]   for 1 <= i, j < L

Row r only depends on row r-1 (element j-1), so the rows form a
dependency chain while each row's j-loop is fully parallel.  We run the
rows in numba's parallel prange (canonical prange(0, n) form): each row
waits on a flag set by the previous row, which turns the serial chain
into a wavefront pipelined across all threads.  The per-row computation
is a streaming vectorized loop, keeping the previous row in cache.
"""
import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def _serial(aa, bb, L):
    for i in range(1, L):
        for j in range(1, L):
            aa[i, j] = aa[i - 1, j - 1] + bb[i, j]


# Loaded through a non-inlinable helper so the compiler cannot hoist the
# load out of the spin loop (it would then be a stale one-shot check).
@nb.njit(nogil=True, inline='never')
def _flag(flags, i):
    return flags[i]


@nb.njit(parallel=True, fastmath=True)
def _par(aa, bb, L, flags):
    n = L - 1
    for k in nb.prange(n):
        r = k + 1
        while _flag(flags, r - 1) < 1:
            pass
        for j in range(1, L):
            aa[r, j] = aa[r - 1, j - 1] + bb[r, j]
        flags[r] = 1


def s119(aa, bb, LEN_2D):
    L = int(LEN_2D)
    if L <= 256:
        _serial(aa, bb, L)
    else:
        flags = np.zeros(L, dtype=np.int64)
        flags[0] = 1
        _par(aa, bb, L, flags)
    return None


def _warm():
    L = 320
    aa = np.zeros((L, L))
    bb = np.zeros((L, L))
    _serial(aa, bb, 16)
    flags = np.zeros(L, dtype=np.int64)
    flags[0] = 1
    _par(aa, bb, L, flags)
    _par(np.asfortranarray(aa), np.asfortranarray(bb), L, flags)
    f32 = np.zeros((64, 64), dtype=np.float32)
    _serial(f32, f32, 64)
    f32b = np.zeros((320, 320), dtype=np.float32)
    fl32 = np.zeros(320, dtype=np.int64)
    fl32[0] = 1
    _par(f32b, f32b, 320, fl32)


_warm()
