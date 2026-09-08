"""Optimized tsvc_2 s3112: inclusive prefix sum  b[i] = a[0] + ... + a[i].

Blocked parallel scan (numba prange, two passes: local block scan, then
parallel add of exclusive block offsets).  Pass 1 software-pipelines four
independent loads ahead of the serial add chain to hide DRAM latency.
Small n falls back to a serial numba loop.  Everything numba is compiled
and warmed at import time (untimed).
"""
import os

import numpy as np

try:
    from numba import njit, prange
    import numba as _numba
    _NUMBA = True
except Exception:  # pragma: no cover
    _NUMBA = False


def _serial(a, b):
    n = a.shape[0]
    s = 0.0
    for i in range(n):
        s += a[i]
        b[i] = s


def _blocked(a, b, nb):
    n = a.shape[0]
    B = (n + nb - 1) // nb
    if B < 1:
        B = 1
    nb2 = (n + B - 1) // B
    part = np.empty(nb2)
    for blk in prange(nb2):
        s = 0.0
        st = blk * B
        en = st + B
        if en > n:
            en = n
        lim = en - ((en - st) & 3)
        for j in range(st, lim, 4):
            x0 = a[j]
            x1 = a[j + 1]
            x2 = a[j + 2]
            x3 = a[j + 3]
            s += x0
            b[j] = s
            s += x1
            b[j + 1] = s
            s += x2
            b[j + 2] = s
            s += x3
            b[j + 3] = s
        for j in range(lim, en):
            s += a[j]
            b[j] = s
        part[blk] = s
    off = 0.0
    for i in range(nb2):
        t = part[i]
        part[i] = off
        off += t
    for blk in prange(nb2):
        p = part[blk]
        st = blk * B
        en = st + B
        if en > n:
            en = n
        for j in range(st, en):
            b[j] += p


_serial_njit = None
_blocked_njit = None
_HAS_PAR = False

if _NUMBA:
    try:
        _serial_njit = njit(cache=False)(_serial)
        _a = np.random.RandomState(0).rand(1 << 20)
        _b = np.zeros(1 << 20)
        _serial_njit(_a, _b)
        assert np.allclose(_b, np.cumsum(_a), rtol=1e-12, atol=1e-9)
    except Exception:
        _serial_njit = None
    try:
        _blocked_njit = njit(parallel=True, cache=False)(_blocked)
        _sm = np.random.RandomState(1).rand(1 << 16)
        _sb = np.empty(1 << 16)
        _blocked_njit(_sm, _sb, 16)
        assert np.allclose(_sb, np.cumsum(_sm), rtol=1e-10, atol=1e-8)
        _HAS_PAR = True
    except Exception:
        _blocked_njit = None

_BLOCKS_PER_THREAD = 8
_SMALL = 1 << 15


def s3112(a, b, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    if n != a.shape[0]:
        a = a[:n]
        b = b[:n]
    if n < _SMALL:
        if _serial_njit is not None:
            _serial_njit(a, b)
        else:
            b[...] = np.cumsum(a)
        return None
    if _HAS_PAR:
        try:
            t = _numba.get_num_threads()
        except Exception:
            t = os.cpu_count() or 4
        nb = t * _BLOCKS_PER_THREAD
        if nb > n:
            nb = n
        if nb < 1:
            nb = 1
        _blocked_njit(a, b, nb)
    else:
        b[...] = np.cumsum(a)
    return None


tsvc_2_s3112_fp64 = s3112
tsvc_2_s3112 = s3112
