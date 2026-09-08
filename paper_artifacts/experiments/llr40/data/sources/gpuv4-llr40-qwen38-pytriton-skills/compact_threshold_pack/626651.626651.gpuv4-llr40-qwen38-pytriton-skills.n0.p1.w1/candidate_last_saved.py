import os
import numpy as np
_T = len(os.sched_getaffinity(0)) or os.cpu_count()
os.environ['NUMBA_NUM_THREADS'] = str(_T)
os.environ['OMP_NUM_THREADS'] = str(_T)
import numba.np.ufunc.parallel as _npar
_npar.set_num_threads(_T)
from numba import njit, prange
B = 65536

@njit(parallel=True)
def kcopy(src, dst, N):
    for i in prange(N):
        dst[i] = src[i] + 1.0

@njit(parallel=True)
def p1(src, m, cnt, N):
    nb = (N + B - 1) // B
    for b in prange(nb):
        lo = b * B; hi = lo + B
        if hi > N: hi = N
        c = 0
        for i in range(lo, hi):
            if src[i] > 0.0:
                c += 1
            m[i] = src[i] > 0.0
        cnt[b] = c

@njit(parallel=True)
def p2(src, weight, packed, m, off, N):
    nb = (N + B - 1) // B
    for b in prange(nb):
        lo = b * B; hi = lo + B
        if hi > N: hi = N
        cur = off[b]
        for i in range(lo, hi):
            if m[i]:
                packed[cur] = src[i] * weight[i]; cur += 1

@njit(parallel=True)
def p2r(src, weight, packed, off, N):
    nb = (N + B - 1) // B
    for b in prange(nb):
        lo = b * B; hi = lo + B
        if hi > N: hi = N
        cur = off[b]
        for i in range(lo, hi):
            if src[i] > 0.0:
                packed[cur] = src[i] * weight[i]; cur += 1

def compact_threshold_pack(src, weight, packed, out_count, LEN_1D):
    import time as _t
    N = LEN_1D
    t0 = _t.perf_counter()
    m = np.empty(N, np.bool_)
    nb = (N + B - 1) // B
    cnt = np.empty(nb, np.int64)
    p1(src, m, cnt, N)
    t1 = _t.perf_counter()
    off = np.empty(nb, np.int64)
    np.cumsum(cnt, out=off); off -= cnt
    p2(src, weight, packed, m, off, N)
    t2 = _t.perf_counter()
    p2r(src, weight, packed, off, N)
    t3 = _t.perf_counter()
    out_count[0] = int(cnt.sum())
    dst = np.empty(N)
    kcopy(src, dst, N)
    t4 = _t.perf_counter()
    kcopy(src, dst, N)
    t5 = _t.perf_counter()
    print(f"N={N} T={_T}: P1 {(t1-t0)*1e3:7.1f}ms P2 {(t2-t1)*1e3:7.1f}ms P2r {(t3-t2)*1e3:7.1f}ms copy {N*16/(t5-t4)/1e9:6.0f} GB/s", flush=True)
    return None

try:
    n0 = 3 * B + 7
    s = np.random.uniform(-1, 1, n0); w = np.random.uniform(-1, 1, n0)
    p = np.zeros(n0); o = np.zeros(1, np.int64)
    compact_threshold_pack(s, w, p, o, n0)
    del s, w, p, o
except Exception as e:
    print("warm fail", e, flush=True)
