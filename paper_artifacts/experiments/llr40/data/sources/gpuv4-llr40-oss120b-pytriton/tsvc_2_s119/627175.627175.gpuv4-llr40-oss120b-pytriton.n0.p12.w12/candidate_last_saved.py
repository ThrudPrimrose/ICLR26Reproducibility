'''Optimized implementation of TSVC_2 kernel s119 using Numba JIT for peak performance.'''

from __future__ import annotations

import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def _s119_jit(aa: np.ndarray, bb: np.ndarray, N: int) -> None:
    for i in range(1, N):
        for j in range(1, N):
            aa[i, j] = aa[i - 1, j - 1] + bb[i, j]

# Warm-up compilation with tiny dummy arrays
_dummy_a = np.zeros((2, 2), dtype=np.float64)
_dummy_b = np.zeros((2, 2), dtype=np.float64)
_s119_jit(_dummy_a, _dummy_b, 2)

def s119(aa: np.ndarray, bb: np.ndarray, LEN_2D: int) -> None:
    # In-place update of aa using the compiled Numba kernel
    _s119_jit(aa, bb, LEN_2D)
