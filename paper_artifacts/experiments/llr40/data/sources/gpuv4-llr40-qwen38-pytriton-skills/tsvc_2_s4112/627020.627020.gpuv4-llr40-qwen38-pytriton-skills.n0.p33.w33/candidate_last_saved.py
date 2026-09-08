"""TSVC tsvc_2 s4112 -- a[i] += b[ip[i]] * 2.0.

Parallel numba, unrolled x4 for memory-level parallelism on the random gather.
The reference loop is DRAM random-gather bound on this 24-core node; the warm
compile at import keeps first-call JIT out of the timed reps.
"""
import numpy as np
from numba import njit, prange

@njit(fastmath=True, parallel=True)
def _k(a, b, ip, N):
    n4 = N // 4
    for blk in prange(n4):
        i = blk * 4
        a[i]   += b[ip[i]] * 2.0
        a[i+1] += b[ip[i+1]] * 2.0
        a[i+2] += b[ip[i+2]] * 2.0
        a[i+3] += b[ip[i+3]] * 2.0
    for i in range(n4 * 4, N):
        a[i] += b[ip[i]] * 2.0

# Warm compile at import (not timed) for both plausible index dtypes.
for _dt in (np.int32, np.int64):
    _k(np.zeros(1024), np.zeros(1024), np.zeros(1024, dtype=_dt), 1024)

def s4112(a, b, ip, LEN_1D):
    _k(a, b, ip, LEN_1D)
