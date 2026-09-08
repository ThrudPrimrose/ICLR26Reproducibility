import os, time
import numpy as np
import numba as nb
from numba import prange

@nb.njit(parallel=True, fastmath=True)
def _copy2(b, s, N):
    for i in prange(N):
        for j in range(N):
            b[i, j] = s[i, j] + 1.0

def fuse_move_ifs(a, b, src, cond, LEN_2D, K):
    N = LEN_2D
    # warm copy
    _copy2(b, src, N)
    t0 = time.perf_counter(); _copy2(b, src, N); t_par = time.perf_counter() - t0
    t0 = time.perf_counter(); b += src; t_ser = time.perf_counter() - t0
    gbps_par = N * N * 8 * 2 / t_par / 1e9
    gbps_ser = N * N * 8 * 2 / t_ser / 1e9
    aff = len(os.sched_getaffinity(0))
    raise RuntimeError(f"PROBE2 N={N} K={K} threads={nb.get_num_threads()} aff={aff} numba_par_copy={t_par*1e3:.1f}ms ({gbps_par:.0f}GB/s) npy_ser={t_ser*1e3:.1f}ms ({gbps_ser:.0f}GB/s)")
