# Parallel affine scan: y[i] = c[i]*y[i-1] + x[i], in-place on y.
# Two-phase blocked scan over the affine monoid, numba prange, all available CPUs.
import os
import numpy as np
import numba
from numba import njit, prange


def _affinity():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return 1


_K = _affinity()
try:
    numba.config.NUMBA_NUM_THREADS = _K
except Exception:
    pass

THRESH = 65536  # below this the serial path wins


@njit
def _serial(y, c, x):
    n = len(x)
    for i in range(1, n):
        y[i] = c[i] * y[i - 1] + x[i]


@njit(parallel=True)
def _pass1(c, x, aj, bj, b):
    m = len(aj)
    n = len(x)
    for j in prange(m):
        s = j * b
        e = s + b
        if e > n:
            e = n
        a = 1.0
        bv = 0.0
        for i in range(s, e):
            a = c[i] * a
            bv = c[i] * bv + x[i]
        aj[j] = a
        bj[j] = bv


@njit
def _pass2(aj, bj, v):
    m = len(aj)
    t = 0.0
    for j in range(m):
        v[j] = t
        t = aj[j] * t + bj[j]


@njit(parallel=True)
def _pass3(c, x, y, v, b):
    m = len(v)
    n = len(x)
    for j in prange(m):
        s = j * b
        e = s + b
        if e > n:
            e = n
        t = v[j]
        for i in range(s, e):
            t = c[i] * t + x[i]
            y[i] = t


def _block_size(n, k):
    import math
    if n <= 0:
        return 1
    target = n // max(1, 4 * k)
    p = 10
    while p < 16 and (1 << p) < target:
        p += 1
    return 1 << p


_bufs = {}


def _get_bufs(m):
    b = _bufs.get(m)
    if b is None:
        b = (np.empty(m), np.empty(m), np.empty(m))
        _bufs[m] = b
    return b


def scan_affine_decay(y, c, x, LEN_1D):
    n = int(LEN_1D)
    if n <= 1:
        return None
    y = np.ascontiguousarray(y)
    c = np.ascontiguousarray(c)
    x = np.ascontiguousarray(x)
    if n < THRESH:
        _serial(y, c, x)
        return None
    b = _block_size(n, _K)
    m = (n + b - 1) // b
    aj, bj, v = _get_bufs(m)
    _pass1(c, x, aj, bj, b)
    _pass2(aj, bj, v)
    _pass3(c, x, y, v, b)
    return None


def _warm():
    n = 1 << 20
    c = np.linspace(0.5, 0.95, n)
    x = np.ones(n)
    y = np.zeros(n)
    b = _block_size(n, _K)
    m = (n + b - 1) // b
    aj, bj, v = _get_bufs(m)
    _serial(y, c, x)
    _pass1(c, x, aj, bj, b)
    _pass2(aj, bj, v)
    _pass3(c, x, y, v, b)


_warm()
