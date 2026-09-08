'''Optimized Python implementation of the wf_triangular benchmark using Numba JIT.

The reference implementation (see /shared/tasks/wf_triangular/wf_triangular_numpy.py)\nuses two nested Python for loops and is therefore very slow for large matrices.
\nThis version attempts to match or exceed the speed of the provided Numba baseline.\nIf Numba is available it compiles a low‑level loop that performs the same\ntriangular wavefront update as the reference, but runs entirely in compiled code\nwith fast‑math optimizations.  If Numba cannot be imported the implementation\nfalls back to a pure‑NumPy vectorised version (still faster than pure Python).\n''' 
from __future__ import annotations

import numpy as np

# Try to import Numba; if unavailable we will use the NumPy fallback.
try:
    import numba
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:
    @numba.njit(fastmath=True)
    def _wf_triangular_numba(a: np.ndarray) -> None:
        N = a.shape[0]
        for i in range(1, N):
            for j in range(i, N):
                a[i, j] = a[i, j] + a[i - 1, j] + a[i, j - 1]
else:
    # Pure‑NumPy fallback – identical to the earlier vectorised version.
    def _wf_triangular_numpy(a: np.ndarray, LEN_2D: int) -> None:
        for i in range(1, LEN_2D):
            base = a[i, i - 1]
            t = a[i, i:LEN_2D] + a[i - 1, i:LEN_2D]
            a[i, i:LEN_2D] = base + np.cumsum(t, dtype=a.dtype)


def wf_triangular(a: np.ndarray, LEN_2D: int) -> None:
    '''Triangular north+west wavefront over j >= i.

    Parameters
    ----------
    a : np.ndarray
        2‑D square array of shape (LEN_2D, LEN_2D). Modified in‑place.
    LEN_2D : int
        The size of each matrix dimension.
    '''
    if LEN_2D <= 1:
        return
    # Ensure we operate on a C‑contiguous array for both Numba and NumPy paths.
    if not a.flags['C_CONTIGUOUS']:
        # Work on a temporary contiguous copy and copy back the result.
        a_tmp = np.ascontiguousarray(a)
        if HAVE_NUMBA:
            _wf_triangular_numba(a_tmp)
        else:
            _wf_triangular_numpy(a_tmp, LEN_2D)
        a[:] = a_tmp
    else:
        if HAVE_NUMBA:
            _wf_triangular_numba(a)
        else:
            _wf_triangular_numpy(a, LEN_2D)
