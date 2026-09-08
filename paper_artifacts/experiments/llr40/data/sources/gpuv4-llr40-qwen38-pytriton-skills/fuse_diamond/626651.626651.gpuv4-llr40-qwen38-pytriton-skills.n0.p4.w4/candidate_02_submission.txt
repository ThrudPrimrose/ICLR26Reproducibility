"""Optimized fuse_diamond: out[i] = (a[i]^2 + 1)(a[i]^2 - 1).

The reference makes 4 passes over the data (t, u, v temporaries); this
fuses everything into a single pass over `a` writing `out` directly.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=False)
def _fused_par(out, a, n):
    for i in prange(n):
        t = a[i] * a[i]
        out[i] = (t + 1.0) * (t - 1.0)


@njit(cache=False)
def _fused_ser(out, a, n):
    for i in range(n):
        t = a[i] * a[i]
        out[i] = (t + 1.0) * (t - 1.0)


def fuse_diamond(out, a, LEN_1D):
    n = int(LEN_1D)
    if n > 1024:
        _fused_par(out, a, n)
    else:
        _fused_ser(out, a, n)
    return None


# Warm up at import time: both variants compile before the timer starts.
_w = np.zeros(1 << 16)
_o = np.zeros(1 << 16)
_fused_ser(_o, _w, _w.size)
_fused_par(_o, _w, _w.size)
