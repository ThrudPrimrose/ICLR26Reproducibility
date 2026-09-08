"""Optimized triangular north+west wavefront: a[i,j] += a[i-1,j] + a[i,j-1] for j >= i.

The parallel front is the anti-diagonal i+j = const, so the whole region can be
computed wavefront by wavefront with full parallelism inside each anti-diagonal.
"""
import os

import numpy as np

# Pin OMP threads to the visible CPUs before numba's OpenMP runtime starts.
if not os.environ.get("OMP_NUM_THREADS"):
    try:
        os.environ["OMP_NUM_THREADS"] = str(len(os.sched_getaffinity(0)))
    except Exception:
        pass

from numba import njit, prange  # noqa: E402


@njit(fastmath=False)
def _wf_ser(f, n):
    """Serial reference-exact loop over a flat C-order buffer."""
    for i in range(1, n):
        r = i * n
        for j in range(i, n):
            f[r + j] = f[r + j] + f[r - n + j] + f[r + j - 1]


@njit(parallel=True, fastmath=False)
def _wf_par(f, n):
    """Parallel wavefront: anti-diagonal d = i+j, all its cells independent."""
    for d in prange(2, 2 * n - 1):
        i0 = d - (n - 1)
        if i0 < 1:
            i0 = 1
        i1 = d // 2
        if i1 > n - 1:
            i1 = n - 1
        for i in range(i0, i1 + 1):
            j = d - i
            r = i * n
            f[r + j] = f[r + j] + f[r - n + j] + f[r + j - 1]


@njit(fastmath=False)
def _wf_ser2d(a, n):
    """Fallback for non-contiguous 2D float64 views."""
    for i in range(1, n):
        for j in range(i, n):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]


_THRESHOLD = 256


def wf_triangular(a, LEN_2D):
    n = int(LEN_2D)
    if a.dtype == np.float64 and a.ndim == 2 and a.flags["C_CONTIGUOUS"]:
        f = a.reshape(-1)
        if n <= _THRESHOLD:
            _wf_ser(f, n)
        else:
            _wf_par(f, n)
        return None
    # Generic fallback: any strided 2D numeric array (keeps in-place contract).
    _wf_ser2d(a, n)
    return None


# --- import-time warmup: compile both code paths before the clock starts ---
def _warm():
    w = np.zeros((64, 64))
    _wf_ser(w.reshape(-1), 64)
    m = _THRESHOLD + 8
    w2 = np.zeros((m, m))
    _wf_par(w2.reshape(-1), m)
    w3 = np.zeros((8, 8), order="F")
    _wf_ser2d(w3, 8)


_warm()
del _warm
