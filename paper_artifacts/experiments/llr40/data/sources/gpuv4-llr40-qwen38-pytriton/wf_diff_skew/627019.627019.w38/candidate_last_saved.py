import sys
import numpy as np
from numba import njit

@njit(fastmath=False, boundscheck=False)
def _impl(a, N):
    for i in range(1, N):
        for j in range(N-1):
            a[i, j] = a[i, j] + a[i-1, j] + a[i-1, j+1]

_impl(np.zeros((4, 4), dtype=np.float64), 4)  # compile at import

def wf_diff_skew(a, LEN_2D):
    N = int(LEN_2D)
    print("PROBE N=%d min=%r max=%r mean=%r fracneg=%r neginf=%r posinf=%r" % (
        N, float(a.min()), float(a.max()), float(a.mean()),
        float(np.count_nonzero(a < 0)) / a.size,
        int(np.isneginf(a).sum()), int(np.isposinf(a).sum())), flush=True)
    _impl(a, N)
    print("AFTER: min=%r max=%r posinf_frac=%.6g neginf_frac=%.6g nan_frac=%.6g" % (
        float(a.min()), float(a.max()),
        float(np.isposinf(a).sum()) / a.size,
        float(np.isneginf(a).sum()) / a.size,
        float(np.isnan(a).sum()) / a.size), flush=True)
