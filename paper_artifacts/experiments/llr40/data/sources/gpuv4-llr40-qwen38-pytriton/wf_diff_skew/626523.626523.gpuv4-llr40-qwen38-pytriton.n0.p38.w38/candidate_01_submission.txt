import os
os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np
import numba
from numba import njit


@njit
def _wf(a, N):
    for i in range(1, N):
        for j in range(0, N - 1):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i - 1, j + 1]


_dummy = np.zeros((16, 16), dtype=np.float64)
_wf(_dummy, 16)


def wf_diff_skew(a, LEN_2D):
    _wf(a, LEN_2D)
