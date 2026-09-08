"""Optimized implementation of wf_diff_skew kernel using a straightforward Numba JIT loop.

The kernel computes:
    a[i, j] = a[i, j] + a[i-1, j] + a[i-1, j+1]

A simple double loop compiled with Numba (``fastmath`` and ``nogil``) gives a very
compact implementation that runs close to native C speed and beats the baseline
reference.  The function mutates ``a`` in place and returns ``None``.
"""

from __future__ import annotations
import numpy as np
from numba import njit

@njit(fastmath=True, nogil=True, boundscheck=False)
def wf_diff_skew(a: np.ndarray, LEN_2D: int) -> None:
    for i in range(1, LEN_2D):
        for j in range(LEN_2D - 1):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i - 1, j + 1]
