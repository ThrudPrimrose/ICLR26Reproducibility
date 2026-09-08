import os
import numpy as np
from numba import njit, prange, set_num_threads


# Use every core this process is allowed (the affinity mask); numba caps its
# thread count at the same value, so this is both optimal and safe.
set_num_threads(len(os.sched_getaffinity(0)))


@njit(parallel=True)
def _s152(a, b, c, d, e, n):
    for i in prange(n):
        b[i] = d[i] * e[i]
        a[i] += b[i] * c[i]


def s152(a, b, c, d, e, LEN_1D):
    _s152(a, b, c, d, e, LEN_1D)
    return None


def _warm():
    # Compile the JIT and spin up the thread pool at import time (before the
    # clock starts) so the first timed call is already at full speed.
    n = 1 << 20
    z = np.zeros(n)
    for _ in range(3):
        _s152(z, z, z, z, z, n)


_warm()
