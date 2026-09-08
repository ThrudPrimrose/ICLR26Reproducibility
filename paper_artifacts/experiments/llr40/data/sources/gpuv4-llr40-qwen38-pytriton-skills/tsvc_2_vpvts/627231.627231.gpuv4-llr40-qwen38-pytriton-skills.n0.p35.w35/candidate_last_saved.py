import os

os.environ["NUMBA_NUM_THREADS"] = "384"

import numpy as np
import numba as _numba
from numba import njit as _njit
from numba import prange as _prange


@_njit(parallel=True, fastmath=True, nogil=True)
def _vpvts_par(a, b, n, S):
    for i in _prange(n):
        a[i] += b[i] * S


def _warm():
    n64 = 4096
    n32 = np.int32(4096)
    s64 = 7
    s32 = np.int32(7)
    a = np.zeros(n64, dtype=np.float64)
    b = np.zeros(n64, dtype=np.float64)
    _vpvts_par(a, b, n64, s64)       # (i64, i64)  -- python ints
    _vpvts_par(a, b, n64, s32)       # (i64, i32)
    _vpvts_par(a, b, n32, s64)       # (i32, i64)
    _vpvts_par(a, b, n32, s32)       # (i32, i32)
    a32 = np.zeros(n64, dtype=np.float32)
    b32 = np.zeros(n64, dtype=np.float32)
    _vpvts_par(a32, b32, n64, s64)   # float32


_warm()


def vpvts(a, b, LEN_1D, S):
    _vpvts_par(a, b, int(LEN_1D), S)
    return None
