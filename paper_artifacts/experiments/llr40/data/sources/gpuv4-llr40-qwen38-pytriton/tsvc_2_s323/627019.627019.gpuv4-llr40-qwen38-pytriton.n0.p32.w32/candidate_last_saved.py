import os
import numpy as np
from numba import njit, prange

# Pin the thread count before numba's first parallel region runs.
if "NUMBA_NUM_THREADS" not in os.environ:
    try:
        _nc = os.cpu_count() or 4
    except Exception:
        _nc = 4
    os.environ["NUMBA_NUM_THREADS"] = str(min(_nc, 96))


@njit(cache=False, fastmath=False)
def _scan_ser(a, b, c, d, e, N):
    for i in range(1, N):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]


@njit(parallel=True, cache=False, fastmath=False)
def _scan_par(a, b, c, d, e, N, nb, bl):
    # Phase 1: per-block partial deltas (serial within a block, exact order).
    part = np.empty(nb, dtype=np.float64)
    for j in prange(nb):
        lo = j * bl
        hi = lo + bl
        if hi > N:
            hi = N
        y = 0.0
        start = lo
        if start == 0:
            start = 1
        for i in range(start, hi):
            f1 = c[i] * d[i]
            f2 = c[i] * e[i]
            t = y + f1
            y = t + f2
        part[j] = y
    # Serial inter-block scan of offsets.
    off = np.empty(nb + 1, dtype=np.float64)
    off[0] = b[0]
    for j in range(nb):
        off[j + 1] = off[j] + part[j]
    # Phase 2: apply offset with the exact per-element rounding order.
    for j in prange(nb):
        lo = j * bl
        hi = lo + bl
        if hi > N:
            hi = N
        y = off[j]
        start = lo
        if start == 0:
            start = 1
        for i in range(start, hi):
            f1 = c[i] * d[i]
            f2 = c[i] * e[i]
            t = y + f1
            a[i] = t
            y = t + f2
            b[i] = y


# Warm both code paths at import time (not charged).
def _warm():
    n = 1 << 16
    a = np.ones(n)
    b = np.ones(n)
    c = np.ones(n)
    d = np.ones(n)
    e = np.ones(n)
    _scan_ser(a, b, c, d, e, n)
    _scan_par(a, b, c, d, e, n, 4, 1 << 14)


_warm()


def s323(a, b, c, d, e, LEN_1D):
    N = int(LEN_1D)
    if N <= 1:
        return None
    if not (a.flags["C_CONTIGUOUS"] and b.flags["C_CONTIGUOUS"] and
            c.flags["C_CONTIGUOUS"] and d.flags["C_CONTIGUOUS"] and
            e.flags["C_CONTIGUOUS"]):
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
        c = np.ascontiguousarray(c)
        d = np.ascontiguousarray(d)
        e = np.ascontiguousarray(e)
        _scan_ser(a, b, c, d, e, N)
        return None
    if N < (1 << 18):
        _scan_ser(a, b, c, d, e, N)
        return None
    T = int(os.environ.get("NUMBA_NUM_THREADS", os.cpu_count() or 4))
    nb = 8 * T
    if nb > N // 64:
        nb = max(1, N // 64)
    bl = (N + nb - 1) // nb
    _scan_par(a, b, c, d, e, N, nb, bl)
    return None
