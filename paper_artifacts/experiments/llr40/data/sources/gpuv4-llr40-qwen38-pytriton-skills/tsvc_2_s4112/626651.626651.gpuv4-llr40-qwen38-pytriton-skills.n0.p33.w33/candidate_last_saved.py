"""s4112 optimized: a[i] += b[ip[i]] * 2.0

Single fused parallel pass (numba prange), in-place ABI (mutates `a`, returns None).
All JIT compilation and threading-layer initialization happen at import time (untimed).
"""
import os

import numpy as np
import numba
from numba import prange


@numba.njit(parallel=True, fastmath=True)
def _s4112(a, b, ip, n):
    for i in prange(n):
        a[i] += b[ip[i]] * 2.0


def _setup_threads():
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = os.cpu_count() or 1
    ncpu = max(1, min(int(ncpu), 64))
    numba.set_num_threads(ncpu)


def _warm():
    _setup_threads()
    _n = 8192
    a = np.zeros(_n, dtype=np.float64)
    b = np.zeros(_n, dtype=np.float64)
    ip = np.arange(_n, dtype=np.int32)
    _s4112(a, b, ip, _n)  # compiles kernel + initializes the threading layer


_warm()


def s4112(a, b, ip, LEN_1D):
    n = int(LEN_1D)
    ok_a = a.dtype == np.float64 and a.flags.c_contiguous
    ok_b = b.dtype == np.float64 and b.flags.c_contiguous
    ok_ip = ip.dtype == np.int32 and ip.flags.c_contiguous
    if ok_a and ok_b and ok_ip:
        _s4112(a, b, ip, n)
        return None
    # defensive path: normalize copies, then write back
    aw = np.ascontiguousarray(a, dtype=np.float64)
    bw = b if ok_b else np.ascontiguousarray(b, dtype=np.float64)
    ipw = ip if ok_ip else np.ascontiguousarray(ip, dtype=np.int32)
    if aw is a:
        aw = a.copy()
    _s4112(aw, bw, ipw, n)
    a[...] = aw
    return None
