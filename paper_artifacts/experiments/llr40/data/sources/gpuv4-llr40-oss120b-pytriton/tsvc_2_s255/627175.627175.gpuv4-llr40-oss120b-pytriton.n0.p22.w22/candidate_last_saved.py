"""Optimized implementation of the TSVC s255 kernel.

The reference implementation (see /shared/tasks/tsvc_2_s255/tsvc_2_s255_numpy.py) computes a
circular three‑point average:

    a[i] = (b[i] + b[i‑1] + b[i‑2]) * 0.333

with wrap‑around indexing (i‑1 and i‑2 are taken modulo the array length).

A naïve Python loop is very slow.  This version uses NumPy vectorised operations and
`np.roll` to express the wrap‑around sum in a single expression, achieving a substantial
speed‑up while preserving the in‑place semantics expected by the benchmark harness.
"""

import numpy as np

# pre‑compute the constant factor – the reference uses 0.333 (≈ 1/3)
_FACTOR = 0.333


def s255(a: np.ndarray, b: np.ndarray, LEN_1D: int) -> None:
    """Compute the s255 kernel in‑place.

    Parameters
    ----------
    a : np.ndarray
        Output buffer, shape (LEN_1D,).  The result is written into this array.
    b : np.ndarray
        Input buffer, shape (LEN_1D,).
    LEN_1D : int
        Length of the 1‑D arrays.  The function respects this value even if the
        supplied arrays are larger.
    """
    # Ensure we work only on the requested length – this matches the reference
    # behaviour when the arrays are larger than LEN_1D.
    n = LEN_1D
    # Vectorised circular three‑point sum.
    # np.roll(b, 1) shifts elements right, i.e. element i becomes b[i‑1] (wrap)
    # np.roll(b, 2) gives b[i‑2].
    a[:n] = (b[:n] + np.roll(b, 1)[:n] + np.roll(b, 2)[:n]) * _FACTOR
    # In‑place write – no return value required by the benchmark harness.
    return None

