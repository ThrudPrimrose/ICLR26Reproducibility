"""Fused single-pass implementation of TSVC tsvc_2_5 fuse_stencil_through_transient.

out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])   for i = 1 .. LEN_1D-3

The transient `tmp` of the reference is eliminated by substituting tmp[i+1]:
this reads `a` once and writes `out` once, fully parallel.
"""
import numpy as np
import numba as nb

@nb.njit(parallel=True, fastmath=False, boundscheck=False)
def _fused(out, a, LEN_1D):
    for i in nb.prange(1, LEN_1D - 2):
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2])


def fuse_stencil_through_transient(out, a, LEN_1D):
    _fused(out, a, LEN_1D)
    return None


# Pre-compile at import time (runs before the timer starts) so no timed rep pays JIT.
_fused(np.zeros(4096), np.zeros(4096), 4096)
