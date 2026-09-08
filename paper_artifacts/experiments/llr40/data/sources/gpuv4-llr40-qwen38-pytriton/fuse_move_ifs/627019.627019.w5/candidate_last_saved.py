"""TSVC tsvc_2_5 kernel ``fuse_move_ifs`` -- optimized python arm.

Reference semantics:
    for i: if cond[i] > 0: a[i, :] = src[i, :] * 2.0
    if K > 0: b[:, :] = src[:, :] + 1.0
In-place on ``a`` and ``b``; returns None.

Strategy:
  * tiny LEN_2D (S=64, fuzzed): single-threaded numba kernel (no thread-pool launch cost)
  * large LEN_2D (M/L/XL): numba prange kernel with the two loop nests FUSED per row
    (src row loaded once, reused for both stores) -> parallel over all cores.
Everything is compiled at import time so the timed region pays nothing.
"""
import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def _serial(a, b, src, cond, K):
    n = a.shape[0]
    if K > 0:
        for i in range(n):
            ai = a[i]
            bi = b[i]
            si = src[i]
            if cond[i] > 0.0:
                for j in range(n):
                    ai[j] = si[j] * 2.0
                    bi[j] = si[j] + 1.0
            else:
                for j in range(n):
                    bi[j] = si[j] + 1.0
    else:
        for i in range(n):
            if cond[i] > 0.0:
                ai = a[i]
                si = src[i]
                for j in range(n):
                    ai[j] = si[j] * 2.0


@njit(parallel=True, fastmath=True)
def _parallel(a, b, src, cond, K):
    n = a.shape[0]
    if K > 0:
        for i in prange(n):
            ai = a[i]
            bi = b[i]
            si = src[i]
            if cond[i] > 0.0:
                for j in range(n):
                    ai[j] = si[j] * 2.0
                    bi[j] = si[j] + 1.0
            else:
                for j in range(n):
                    bi[j] = si[j] + 1.0
    else:
        for i in prange(n):
            if cond[i] > 0.0:
                ai = a[i]
                si = src[i]
                for j in range(n):
                    ai[j] = si[j] * 2.0


_SMALL = 256  # serial below, parallel at/above (S=64 serial; M/L/XL parallel)


def fuse_move_ifs(a, b, src, cond, LEN_2D, K):
    if LEN_2D <= _SMALL:
        _serial(a, b, src, cond, K)
    else:
        _parallel(a, b, src, cond, K)
    return None


# ---- import-time warm-up: compile all specializations, pay nothing at call time ----
def _warm():
    n = 128
    a = np.empty((n, n), dtype=np.float64)
    b = np.empty((n, n), dtype=np.float64)
    src = np.empty((n, n), dtype=np.float64)
    cond = np.empty(n, dtype=np.float64)
    src.fill(1.0)
    cond.fill(1.0)
    for K in (1, -1):
        _serial(a, b, src, cond, K)
        _parallel(a, b, src, cond, K)
    # warm the big-row parallel path (same signature, larger n) so no shape-dependent
    # work happens later
    n = 300
    a = np.empty((n, n), dtype=np.float64)
    b = np.empty((n, n), dtype=np.float64)
    src = np.empty((n, n), dtype=np.float64)
    cond = np.empty(n, dtype=np.float64)
    src.fill(1.0)
    cond.fill(1.0)
    _parallel(a, b, src, cond, 1)


_warm()
