"""Optimized TSVC tsvc_2_5 kernel ``fuse_stencil_through_transient``.

Reference (C oracle):
    for i in 1..LEN_1D-3:
        out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])

Each ``out[i]`` depends only on ``a[i-1..i+2]`` -- iterations are independent -- so the
single fused loop (one read pass over ``a``, one write pass over ``out``) is parallelized
over all pinned cores with ``numba.prange`` and a manual 8-way unroll for load-level
parallelism.  A small-size serial path avoids the parallel fork-join cost.  All JIT
compilation happens at import time, before the timed calls.
"""

import os

# Select numba's OpenMP threading layer (lowest, most stable fork-join overhead).
# Must be set before numba is imported.
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import numpy as np
import numba
from numba import njit, prange

__all__ = ["fuse_stencil_through_transient"]

_R = 8
_THRESHOLD = 65536


@njit(parallel=True)
def _par_kernel(out, a):
    n = len(a)
    m = (n - 3) // _R
    if m < 0:
        m = 0
    for c in prange(m):
        i = 1 + c * _R
        out[i]     = (a[i - 1] + a[i]     + a[i + 1]) * (a[i]     + a[i + 1] + a[i + 2])
        out[i + 1] = (a[i]     + a[i + 1] + a[i + 2]) * (a[i + 1] + a[i + 2] + a[i + 3])
        out[i + 2] = (a[i + 1] + a[i + 2] + a[i + 3]) * (a[i + 2] + a[i + 3] + a[i + 4])
        out[i + 3] = (a[i + 2] + a[i + 3] + a[i + 4]) * (a[i + 3] + a[i + 4] + a[i + 5])
        out[i + 4] = (a[i + 3] + a[i + 4] + a[i + 5]) * (a[i + 4] + a[i + 5] + a[i + 6])
        out[i + 5] = (a[i + 4] + a[i + 5] + a[i + 6]) * (a[i + 5] + a[i + 6] + a[i + 7])
        out[i + 6] = (a[i + 5] + a[i + 6] + a[i + 7]) * (a[i + 6] + a[i + 7] + a[i + 8])
        out[i + 7] = (a[i + 6] + a[i + 7] + a[i + 8]) * (a[i + 7] + a[i + 8] + a[i + 9])
    i = 1 + m * _R
    while i < n - 2:
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2])
        i += 1


@njit
def _ser_kernel(out, a):
    n = len(a)
    for i in range(1, n - 2):
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2])


def _set_threads():
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    n = min(n, os.cpu_count() or 1)
    if n >= 1:
        numba.set_num_threads(n)


def _warmup():
    a = np.zeros(1024)
    o = np.zeros(1024)
    _ser_kernel(o, a)
    _par_kernel(o, a)


_set_threads()
_warmup()


def fuse_stencil_through_transient(out, a, LEN_1D):
    """In-place ABI: writes the result into ``out`` (indices 1..LEN_1D-3), returns None."""
    n = LEN_1D
    if out.dtype != a.dtype:
        buf = np.empty(n, dtype=a.dtype)
        if n >= _THRESHOLD:
            _par_kernel(buf, np.ascontiguousarray(a))
        else:
            _ser_kernel(buf, np.ascontiguousarray(a))
        out[...] = buf
        return None
    if a.flags.c_contiguous and out.flags.c_contiguous:
        if n >= _THRESHOLD:
            _par_kernel(out, a)
        else:
            _ser_kernel(out, a)
        return None
    buf = np.empty(n, dtype=a.dtype)
    av = np.ascontiguousarray(a)
    if n >= _THRESHOLD:
        _par_kernel(buf, av)
    else:
        _ser_kernel(buf, av)
    out[...] = buf
    return None
