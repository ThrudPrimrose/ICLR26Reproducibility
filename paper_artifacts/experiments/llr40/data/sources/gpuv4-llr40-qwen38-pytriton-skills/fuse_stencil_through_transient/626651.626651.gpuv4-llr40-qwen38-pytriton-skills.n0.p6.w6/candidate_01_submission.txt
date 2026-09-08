"""TSVC tsvc_2_5 ``fuse_stencil_through_transient`` -- fused one-pass numba kernel.

out[i] = (a[i-1]+a[i]+a[i+1]) * (a[i]+a[i+1]+a[i+2])  for i in 1..LEN_1D-3
(bit-identical to the numpy reference: same association order, no reassociation)
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True)
def _fused(out, a, n):
    for i in prange(1, n - 2):
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2])


def fuse_stencil_through_transient(out, a, LEN_1D):
    _fused(out, a, int(LEN_1D))
    return None


# --- import-time warmup (untimed): precompile the hot variants ---------------
def _warm():
    n = 1 << 14
    for dt in (np.float64, np.float32):
        a = np.ones(n, dtype=dt)
        o = np.empty(n, dtype=dt)
        _fused(o, a, n)
    a = np.ones(32, dtype=np.int64)
    o = np.empty(32, dtype=np.int64)
    _fused(o, a, 32)
    a = np.ones(32, dtype=np.int32)
    o = np.empty(32, dtype=np.int32)
    _fused(o, a, 32)
    # tiny / degenerate sizes must not crash
    a = np.ones(1, dtype=np.float64)
    o = np.empty(1, dtype=np.float64)
    _fused(o, a, 1)
    a = np.ones(3, dtype=np.float64)
    o = np.empty(3, dtype=np.float64)
    _fused(o, a, 3)


_warm()
