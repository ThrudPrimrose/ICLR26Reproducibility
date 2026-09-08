import os, time, statistics
import numpy as np
import numba
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def t_add(a, b, n):
    for i in prange(n):
        b[i] = a[i] + 1.0

@njit(parallel=True, fastmath=True)
def t_copy2(a, b, c, d, n):
    # two independent streams concurrently (like b and a parts)
    for i in prange(n):
        b[i] = a[i] + 1.0
    for i in prange(n):
        d[i] = c[i] * 2.0

GB = 1200 * 1024 * 1024 // 8
a = np.random.RandomState(0).rand(GB)
b = np.empty(GB)
c = np.random.RandomState(1).rand(GB)
d = np.empty(GB)

def bench(fn, n=4):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

for nt in (8, 16, 24):
    numba.set_num_threads(nt)
    mn = bench(lambda: t_add(a, b, GB))
    print(f"add nt={nt}: {mn*1e3:.3f} ms -> {GB*16/mn/1e9:.0f} GB/s")
for nt in (24,):
    numba.set_num_threads(nt)
    mn = bench(lambda: t_copy2(a, b, c, d, GB//2))
    print(f"copy2 nt={nt}: {mn*1e3:.3f} ms -> {GB*16/mn/1e9:.0f} GB/s")
print("PROBE7 DONE")

def fuse_move_ifs(a, b, src, cond, LEN_2D, K):
    m = cond > 0.0
    a[m] = src[m] * 2.0
    if K > 0:
        b[...] = src + 1.0
    return None
