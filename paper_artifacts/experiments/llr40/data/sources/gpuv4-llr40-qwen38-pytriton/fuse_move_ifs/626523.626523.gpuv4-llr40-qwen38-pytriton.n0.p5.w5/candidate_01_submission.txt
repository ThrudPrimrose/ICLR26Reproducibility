"""TSVC tsvc_2_5 fuse_move_ifs -- optimized Python implementation.

Reference semantics (in-place):
    for i in range(LEN_2D):
        if cond[i] > 0.0:
            a[i, :] = src[i, :] * 2.0
    if K > 0:
        b = src + 1.0
Rows of `a` with cond[i] <= 0 and `b` when K <= 0 are left untouched.

Strategy: numba-compiled kernels (JIT compiled at import time, so nothing
slow is charged to the timed calls).  Large sizes use a single fused
parallel region over rows (one read of src, conditional write of a,
write of b); small sizes use the serial kernel to avoid thread overhead.
A vectorized NumPy path covers any dtype/layout the kernels were not
warmed for, so the function is always correct.
"""

import numpy as np
from numba import njit, prange

_PARALLEL_MIN = 800  # below this row count, serial wins (measured)


@njit(parallel=True)
def _kern_p(a, b, src, cond, n, K):
    kp = K > 0
    for i in prange(n):
        cm = cond[i] > 0.0
        if cm and kp:
            for j in range(n):
                v = src[i, j]
                a[i, j] = v * 2.0
                b[i, j] = v + 1.0
        elif cm:
            for j in range(n):
                a[i, j] = src[i, j] * 2.0
        elif kp:
            for j in range(n):
                b[i, j] = src[i, j] + 1.0


@njit
def _kern_s(a, b, src, cond, n, K):
    kp = K > 0
    for i in range(n):
        cm = cond[i] > 0.0
        if cm and kp:
            for j in range(n):
                v = src[i, j]
                a[i, j] = v * 2.0
                b[i, j] = v + 1.0
        elif cm:
            for j in range(n):
                a[i, j] = src[i, j] * 2.0
        elif kp:
            for j in range(n):
                b[i, j] = src[i, j] + 1.0


def _fallback(a, b, src, cond, n, K):
    """Vectorized NumPy reference (covers exotic dtypes/layouts)."""
    mask = cond[:n] > 0.0
    if mask.all():
        np.multiply(src[:n, :n], 2.0, out=a[:n, :n])
    elif mask.any():
        a[:n][mask] = src[:n][mask] * 2.0
    if K > 0:
        np.add(src[:n, :n], 1.0, out=b[:n, :n])


def fuse_move_ifs(a, b, src, cond, LEN_2D, K):
    n = int(LEN_2D)
    dt = a.dtype
    if isinstance(K, (bool, int, np.integer)):
        K = int(K)
    elif isinstance(K, (float, np.floating)):
        K = float(K)
    else:
        _fallback(a, b, src, cond, n, K)
        return None
    if (
        (dt == np.float64 or dt == np.float32)
        and b.dtype == dt
        and src.dtype == dt
        and cond.dtype == dt
        and a.shape == (n, n)
        and b.shape == (n, n)
        and src.shape == (n, n)
        and cond.ndim == 1
        and cond.shape[0] >= n
        and a.flags.c_contiguous
        and b.flags.c_contiguous
        and src.flags.c_contiguous
        and cond.flags.c_contiguous
    ):
        if n >= _PARALLEL_MIN:
            _kern_p(a, b, src, cond, n, K)
        else:
            _kern_s(a, b, src, cond, n, K)
    else:
        _fallback(a, b, src, cond, n, K)
    return None


def _warm():
    n = 32
    half = n // 2
    for dt in (np.float64, np.float32):
        a = np.zeros((n, n), dtype=dt)
        b = np.zeros((n, n), dtype=dt)
        s = np.ones((n, n), dtype=dt)
        c = np.ones(n, dtype=dt)
        c[:half] = -1.0  # exercise mixed mask branches
        for kk in (1, 0, 1.0):
            _kern_s(a, b, s, c, n, kk)
            _kern_p(a, b, s, c, n, kk)


_warm()
