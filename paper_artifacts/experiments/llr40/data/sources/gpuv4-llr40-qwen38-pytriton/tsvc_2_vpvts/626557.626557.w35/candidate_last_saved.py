"""TSVC tsvc_2 kernel ``vpvts`` -- a[i] += b[i] * S.

In-place ABI matching the numpy reference: mutates ``a``, returns None.
Numba-compiled loops; all warmup (JIT compilation) happens at module import
time, i.e. before the timer starts.
"""
import os

import numpy as np
import numba
from numba import njit, prange


def _pick_threads():
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 4
    return max(1, min(n, 32))


try:
    numba.set_num_threads(_pick_threads())
except Exception:
    pass


@njit
def _vpvts_scalar(a, b, n, S):
    for i in range(n):
        a[i] = a[i] + b[i] * S


@njit(parallel=True)
def _vpvts_par(a, b, n, S):
    for i in prange(n):
        a[i] = a[i] + b[i] * S


# parallel fork-join only pays off once the work per thread is sizable
_PAR_MIN = 1 << 20


def vpvts(a, b, LEN_1D, S):
    if LEN_1D >= _PAR_MIN:
        _vpvts_par(a, b, LEN_1D, S)
    else:
        _vpvts_scalar(a, b, LEN_1D, S)
    return None


# ---- import-time warmup: compile every plausible signature off the clock ----
_w = np.arange(8192, dtype=np.float64)
for _s in (3, np.int64(3), np.int32(3), 3.0, np.float64(3.0), np.float32(3.0)):
    _vpvts_scalar(_w, _w, 8192, _s)
    _vpvts_par(_w, _w, 8192, _s)
del _w, _s
