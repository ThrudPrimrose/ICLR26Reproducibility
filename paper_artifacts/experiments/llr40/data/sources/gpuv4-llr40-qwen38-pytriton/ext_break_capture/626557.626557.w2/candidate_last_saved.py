import os

_ncpu = os.cpu_count() or 8
_T = min(96, _ncpu)
os.environ["NUMBA_NUM_THREADS"] = str(_T)

import numpy as np
from numba import njit, prange


@njit(fastmath=True, nogil=True, cache=True)
def _serial(a, K, out_index, out_value):
    n = a.shape[0]
    out_index[0] = -1
    out_value[0] = -1.0
    for i in range(n):
        if a[i] > K:
            out_index[0] = i
            out_value[0] = a[i]
            break


@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def _parallel(a, K, out_index, out_value, nblocks, block):
    n = a.shape[0]
    first = np.full(nblocks, -1, dtype=np.int64)
    for b in prange(nblocks):
        lo = b * block
        hi = lo + block
        if hi > n:
            hi = n
        f = -1
        for i in range(lo, hi):
            if a[i] > K:
                f = i
                break
        first[b] = f
    bi = -1
    for b in range(nblocks):
        if first[b] != -1:
            bi = b
            break
    if bi < 0:
        out_index[0] = -1
        out_value[0] = -1.0
    else:
        i = first[bi]
        out_index[0] = i
        out_value[0] = a[i]


_SERIAL_MAX = 1 << 21
_BLOCK = 1 << 21


def ext_break_capture(a, out_index, out_value, LEN_1D, K,
                      _s=_serial, _p=_parallel):
    n = int(LEN_1D)
    if n != a.shape[0]:
        a = a[:n]
    if a.dtype != np.dtype(np.float64) or not a.flags.c_contiguous:
        a = np.ascontiguousarray(a, dtype=np.float64)
    K = float(K)
    if n <= _SERIAL_MAX:
        _s(a, K, out_index, out_value)
    else:
        nblocks = (n + _BLOCK - 1) // _BLOCK
        _p(a, K, out_index, out_value, nblocks, _BLOCK)


def _warm():
    oi = np.zeros(1, dtype=np.int64)
    ov = np.zeros(1, dtype=np.float64)
    a = np.zeros(4096, dtype=np.float64)
    a[100] = 777.0
    _serial(a, 1.0, oi, ov)
    _parallel(a, 1.0, oi, ov, 4, 1024)
    b = np.zeros(8, dtype=np.float64)
    _serial(b, 1.0, oi, ov)
    _parallel(b, 1.0, oi, ov, 2, 4)


_warm()
