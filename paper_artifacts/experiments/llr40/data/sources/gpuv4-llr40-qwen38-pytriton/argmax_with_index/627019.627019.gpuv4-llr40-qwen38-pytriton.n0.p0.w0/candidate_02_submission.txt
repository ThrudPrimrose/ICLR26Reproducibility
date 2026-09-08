"""TSVC argmax_with_index: running maximum carrying value and 0-based index.

Strategy: single pass over the array. For large inputs, split into chunks and
run np.argmax (which releases the GIL) on each chunk from a thread pool, then
combine with strict '>' in chunk order (first occurrence wins, matching the
reference). The pool and the thread count are calibrated once at import time.
"""
import os
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor

_CAL_N = 128 * 1024 * 1024 // 8   # 128 MiB of fp64
_MIN_CHUNK = 1024 * 1024          # 8 MiB of fp64 per chunk
_SINGLE_BELOW = 3 * 1024 * 1024   # below ~24 MiB stay single-threaded


def _pooled_argmax(a, ex, K):
    n = a.shape[0]
    ch = (n + K - 1) // K
    fs = [ex.submit(np.argmax, a[s * ch:min(n, (s + 1) * ch)]) for s in range(K)]
    r = [f.result() for f in fs]
    bv = -np.inf
    bs = 0
    for s in range(K):
        v = a[s * ch + r[s]]
        if v > bv:
            bv = v
            bs = s
    return bs * ch + r[bs]


def _calibrate():
    ex = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4))
    a = np.arange(_CAL_N, dtype=np.float64)
    best = 1
    best_t = 1e30
    for K in (1, 2, 4, 6, 8, 12, 16, 24, 32):
        t = 1e30
        for _ in range(2):
            t0 = time.perf_counter()
            _pooled_argmax(a, ex, K)
            t = min(t, time.perf_counter() - t0)
        if t < best_t:
            best_t = t
            best = K
    del a
    return ex, best


try:
    _EX, _K_BEST = _calibrate()
except Exception:
    _EX = None
    _K_BEST = 1


def argmax_with_index(a, out_value, out_index, LEN_1D):
    n = LEN_1D
    if _EX is None or n < _SINGLE_BELOW:
        idx = int(np.argmax(a[:n]))
        out_index[0] = idx
        out_value[0] = a[idx]
        return None
    K = min(_K_BEST, n // _MIN_CHUNK)
    if K < 2:
        idx = int(np.argmax(a[:n]))
    else:
        idx = _pooled_argmax(a[:n], _EX, K)
    out_index[0] = idx
    out_value[0] = a[idx]
    return None
