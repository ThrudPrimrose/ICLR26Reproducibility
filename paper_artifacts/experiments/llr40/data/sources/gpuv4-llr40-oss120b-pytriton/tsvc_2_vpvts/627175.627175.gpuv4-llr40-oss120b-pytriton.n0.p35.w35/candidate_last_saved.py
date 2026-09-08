"""Optimized vpvts kernel using Numba for parallel execution.
\nThe reference implementation uses a Python ``for`` loop and is much slower.\nThis version compiles a simple loop with Numba and enables multi‑core\nparallelism via ``prange``.  The operation is memory‑bandwidth bound, so\nparallel execution reduces wall‑clock time on multi‑core systems.\n"""

import numpy as np
import numba

@numba.njit(parallel=True)
def _vpvts_numba(a, b, S):
    n = a.shape[0]
    for i in numba.prange(n):
        a[i] = a[i] + b[i] * S

def vpvts(a, b, LEN_1D, S):
    """Update ``a`` in place: ``a[i] += b[i] * S`` for ``i`` in ``0..LEN_1D-1``.

    Parameters
    ----------
    a : np.ndarray
        Output array, modified in‑place.
    b : np.ndarray
        Input array of same length as ``a``.
    LEN_1D : int
        Declared length of the arrays.  The function asserts that this matches the\
        actual array sizes; the argument is kept for the benchmark harness signature.
    S : scalar
        Scalar multiplier (int or float).\n    """
    assert a.shape[0] == LEN_1D and b.shape[0] == LEN_1D, (
        f"Length mismatch: LEN_1D={LEN_1D}, a.shape={a.shape}, b.shape={b.shape}")
    _vpvts_numba(a, b, S)
    return None
