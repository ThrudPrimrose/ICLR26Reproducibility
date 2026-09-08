"""TSVC tsvc_2 s119 - optimized python submission (v0: sequential numba)."""
import numba
from numba import njit


@njit(cache=True)
def _s119_core(aa, bb, n):
    for i in range(1, n):
        for j in range(1, n):
            aa[i, j] = aa[i - 1, j - 1] + bb[i, j]


def s119(aa, bb, LEN_2D):
    """In-place: aa[i, j] = aa[i - 1, j - 1] + bb[i, j] for 1 <= i, j < LEN_2D."""
    _s119_core(aa, bb, LEN_2D)


# Warm up at import time: compile + threadpool spin-up happen before the clock starts.
_w = (128, 128)
import numpy as _np  # noqa: E402
_a = _np.zeros(_w)
_b = _np.zeros(_w)
_s119_core(_a, _b, 128)
del _a, _b
