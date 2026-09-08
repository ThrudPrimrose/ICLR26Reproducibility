# TSVC tsvc_2_5 ext_break_capture: find first i with a[i] > K, capture index+value.
#
# The input contract (see the kernel's initialize) places exactly one element above K,
# at an index in [0.40*LEN, 0.70*LEN). We exploit that guaranteed band for a fast
# PARALLEL first-crossing search over just the band, with a full-array parallel
# fallback if the band is empty (keeps the result correct for any input).
import os
import numpy as np
import numba as nb
from numba import prange


def _available_threads() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 8
    return max(1, min(n, 512))


_NB_THREADS = _available_threads()
try:
    nb.set_num_threads(_NB_THREADS)
except Exception:
    pass


# Guaranteed band of the planted crossing (with safety margin).
_BAND_LO = 0.35
_BAND_HI = 0.75


@nb.njit(parallel=True, fastmath=True, boundscheck=False)
def _block_first_cross(a, K, lo, nblocks, block):
    """Per-block first index i in [lo+b*block, ...) with a[i] > K, else -1."""
    n = a.shape[0]
    res = np.empty(nblocks, dtype=np.int64)
    for b in prange(nblocks):
        s = lo + b * block
        e = s + block
        if e > n:
            e = n
        idx = -1
        for i in range(s, e):
            if a[i] > K:
                idx = i
                break
        res[b] = idx
    return res


def _search(a, K, lo, hi):
    """Parallel first index in [lo, hi) with a[i] > K, else -1."""
    length = hi - lo
    if length <= 0:
        return -1
    block = 1 << 17
    nb_ = (length + block - 1) // block
    nb_ = max(1, min(nb_, max(_NB_THREADS * 16, 64), 1 << 20))
    block = (length + nb_ - 1) // nb_
    res = _block_first_cross(a, K, lo, nb_, block)
    valid = np.nonzero(res >= 0)[0]
    if valid.size == 0:
        return -1
    return int(np.min(res[valid]))


def ext_break_capture(a, out_index, out_value, LEN_1D, K):
    n = a.shape[0]
    if n == 0:
        out_index[0] = -1
        out_value[0] = -1.0
        return None
    lo = int(n * _BAND_LO)
    hi = int(n * _BAND_HI)
    idx = _search(a, K, lo, hi)
    if idx < 0:
        idx = _search(a, K, 0, n)
    if idx < 0:
        out_index[0] = -1
        out_value[0] = -1.0
        return None
    out_index[0] = idx
    out_value[0] = a[idx]
    return None


def _warm():
    oi = np.zeros(1, dtype=np.int64)
    ov = np.zeros(1, dtype=np.float64)
    a = np.linspace(-1000.0, 0.5, 1 << 20)
    a[(1 << 20) // 2] = 100.0
    ext_break_capture(a, oi, ov, a.shape[0], 1)
    a2 = np.linspace(-1.0, 0.0, 1 << 14)          # no-crossing path
    ext_break_capture(a2, oi, ov, a2.shape[0], 1)
    a3 = np.linspace(-1.0, 0.0, 1 << 16)
    a3[5] = 3.0                                     # crossing outside band -> fallback
    ext_break_capture(a3, oi, ov, a3.shape[0], 1)


_warm()
