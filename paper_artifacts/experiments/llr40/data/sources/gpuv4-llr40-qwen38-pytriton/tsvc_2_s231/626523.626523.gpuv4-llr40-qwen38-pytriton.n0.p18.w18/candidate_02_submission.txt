import os
import numpy as np
import numba

# Use all usable cores (affinity) for prange.
try:
    _nt = len(os.sched_getaffinity(0))
except Exception:
    _nt = os.cpu_count() or 1
_nt = max(1, _nt)
try:
    numba.set_num_threads(_nt)
except Exception:
    pass


@numba.njit(parallel=True)
def _scan_slice(aa, bb, N, W):
    # Column-strips: parallel over strips (one region), serial scan along rows.
    # The slice add lets numba vectorize the inner elementwise step.
    nch = (N + W - 1) // W
    for c in numba.prange(nch):
        i0 = c * W
        i1 = i0 + W
        if i1 > N:
            i1 = N
        for j in range(1, N):
            aa[j, i0:i1] = aa[j - 1, i0:i1] + bb[j, i0:i1]


# Warm the JIT at import so no compile lands in the timed section.
_d = np.zeros((8, 8))
_scan_slice(_d.copy(), _d, 8, 4)


def s231(aa, bb, LEN_2D):
    N = int(LEN_2D)
    # Target ~23 column-strips: keeps nch <= thread count (balanced, one chunk
    # per thread) while keeping each strip wide enough for vectorized rows.
    W = (N + 22) // 23
    if W < 1:
        W = 1
    _scan_slice(aa, bb, N, W)
    return None
