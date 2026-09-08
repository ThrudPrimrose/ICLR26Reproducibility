"""TSVC tsvc_2_5 kernel ``ext_war_unit``:  a[i] = a[i+1] + b[i]  for i in 0..n-2.

The loop carries only an *external WAR* anti-dependence (it reads a[i+1], which a
later iteration writes).  Each output element therefore depends only on the ORIGINAL
inputs, so the whole thing is embarrassingly parallel -- it is a 1-element shift.

Strategy
--------
* small n: the serial numba loop (matches the warmed serial baseline; numba's
  parallel fork/barrier overhead would dominate at this size).
* large n: fully-parallel block decomposition.  Split [0,n) into B contiguous blocks.
  Inside a block the in-block read a[i+1] is safe (sequential within the block: we read
  a[i+1] at step i, before writing it at step i+1).  The ONLY cross-block access is the
  last index of a block reading the first element of the next block; that single value
  per block is pre-captured into a tiny ``seeds`` array before the parallel phase, so
  total memory traffic stays at the irreducible 3 arrays (read a, read b, write a) with
  NO full temporary copy.
"""
import numpy as _np
from numba import njit, prange

_NBLOCKS = 256
_PAR_THRESHOLD = 2_000_000


@njit
def _serial(a, b, n):
    for i in range(n - 1):
        a[i] = a[i + 1] + b[i]


@njit(parallel=True)
def _bseed(a, b, n, B, S):
    seeds = _np.empty(B)
    for j in range(B):
        idx = (j + 1) * S
        seeds[j] = a[idx] if idx < n else 0.0
    for j in prange(B):
        lo = j * S
        hi = (j + 1) * S
        if hi > n - 1:
            hi = n - 1
        for i in range(lo, hi):
            a[i] = a[i + 1] + b[i]
        last = hi - 1
        nb = (j + 1) * S
        if last == nb - 1 and nb < n:
            a[last] = seeds[j] + b[last]


def ext_war_unit(a, b, LEN_1D):
    n = int(LEN_1D)
    if n < 2:
        return None
    if n >= _PAR_THRESHOLD:
        B = _NBLOCKS
        S = (n + B - 1) // B
        _bseed(a, b, n, B, S)
    else:
        _serial(a, b, n)
    return None


def _warm():
    # Force the numba JIT compile + parallel thread-pool init at IMPORT time (not timed).
    n = 1 << 16
    a = _np.ones(n)
    b = _np.ones(n)
    _serial(a, b, n)
    B = _NBLOCKS
    S = (n + B - 1) // B
    _bseed(a, b, n, B, S)


_warm()
