"""
Optimized implementation for TSVC kernel s4112 using Numba.
The reference kernel performs: a[i] = a[i] + b[ip[i]] * 2.0
We use a Numba JIT compiled loop (fastmath) to achieve near‑C speed.
The function updates `a` in‑place and returns None (in‑place ABI).
"""

import numpy as np

# Numba JIT implementation (fast, no Python loops)
try:
    import numba as nb
    _has_numba = True
except Exception:
    _has_numba = False

if _has_numba:
    @nb.njit(parallel=True, fastmath=True)
    def _s4112_numba(a, b, ip):
        for i in nb.prange(a.shape[0]):
            a[i] += b[ip[i]] * 2.0

def s4112(a: np.ndarray, b: np.ndarray, ip: np.ndarray, LEN_1D: int):
    """In‑place update of a according to the TSVC s4112 kernel.

    Parameters
    ----------
    a : np.ndarray (float64)
        Input/output array of length ``LEN_1D``.
    b : np.ndarray (float64)
        Input array of length ``LEN_1D``.
    ip : np.ndarray (int32)
        Index permutation of length ``LEN_1D``.
    LEN_1D : int
        Logical length of the arrays.
    """
    if a.shape != (LEN_1D,) or b.shape != (LEN_1D,) or ip.shape != (LEN_1D,):
        raise ValueError("Input shapes do not match LEN_1D")
    if _has_numba:
        _s4112_numba(a, b, ip)
        return None
    # Fallback: pure NumPy (still faster than the reference loop).
    a += 2.0 * b[ip]
    return None
