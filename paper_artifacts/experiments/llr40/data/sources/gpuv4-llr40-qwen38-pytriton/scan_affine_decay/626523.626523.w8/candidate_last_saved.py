import os
import numpy as np

try:
    _NCPU = len(os.sched_getaffinity(0))
except Exception:
    _NCPU = os.cpu_count() or 1
import numba
import numba.np.unsafe as _nu  # noqa
from numba import njit, prange

numba.set_num_threads(_NCPU)


@njit(fastmath=True, parallel=True)
def _scan3(y, c, x, n, cs, A, B, PA, PB):
    # domain: positions 1..n-1 in chunks of size cs
    m = n - 1
    nb = (m + cs - 1) // cs
    # stage 1: local scans (input value 0), store local results in y, record chunk pairs
    for i in prange(nb):
        s = 1 + i * cs
        e = s + cs
        if e > n:
            e = n
        v = 0.0
        a = 1.0
        for j in range(s, e):
            v = c[j] * v + x[j]
            a = c[j] * a
            y[j] = v
        A[i] = a
        B[i] = v
    # stage 2: exclusive scan over chunk pairs
    x0 = y[0]
    pa = 1.0
    pb = 0.0
    for i in range(nb):
        PA[i] = pa
        PB[i] = pb
        a = A[i]
        pa = a * pa
        pb = a * pb + B[i]
    # stage 3: rescan each chunk with the true incoming value
    for i in prange(nb):
        s = 1 + i * cs
        e = s + cs
        if e > n:
            e = n
        v = PA[i] * x0 + PB[i]
        for j in range(s, e):
            v = c[j] * v + x[j]
            y[j] = v
    return None


def scan_affine_decay(y, c, x, LEN_1D):
    n = int(LEN_1D)
    if n < 2:
        return None
    if not (y.flags.c_contiguous and c.flags.c_contiguous and x.flags.c_contiguous):
        y2 = np.ascontiguousarray(y)
        c2 = np.ascontiguousarray(c)
        x2 = np.ascontiguousarray(x)
        A = np.empty(1, np.float64)
        B = np.empty(1, np.float64)
        PA = np.empty(1, np.float64)
        PB = np.empty(1, np.float64)
        _scan3(y2, c2, x2, n, 1 << 16, A, B, PA, PB)
        y[...] = y2
        return None
    dt = y.dtype
    A = np.empty(n // (1 << 16) + 2, dt)
    B = np.empty_like(A)
    PA = np.empty_like(A)
    PB = np.empty_like(A)
    _scan3(y, c, x, n, 1 << 16, A, B, PA, PB)
    return None


# ---- import-time warm-up (JIT compile before the clock starts) ----
def _warm():
    for dt in (np.float64, np.float32):
        n = 4096
        y = np.zeros(n, dt)
        c = np.full(n, 0.5, dt)
        x = np.ones(n, dt)
        A = np.empty(16, dt)
        B = np.empty(16, dt)
        PA = np.empty(16, dt)
        PB = np.empty(16, dt)
        for _ in range(3):
            _scan3(y, c, x, n, 1 << 16, A, B, PA, PB)
        _scan3(y, c, x, 8, 1 << 16, A, B, PA, PB)


_warm()
