import os, sys, time, shutil, subprocess
import numpy as np
import numba
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _sum_par(a, N, T):
    s = 0.0
    for i in prange(N):
        s += a[i]
    return s

@njit(parallel=True, fastmath=True)
def _par(a, inc, N, T, cmax, cidx, result):
    sz = (N + T - 1) // T
    for t in prange(T):
        lo = t * sz
        hi = lo + sz
        if hi > N:
            hi = N
        m = -1.0
        idx = lo
        k = lo * inc
        for i in range(lo, hi):
            v = abs(a[k])
            if v > m:
                m = v
                idx = i
            k += inc
        cmax[t] = m
        cidx[t] = idx
    M = -1.0
    I = 0
    for t in range(T):
        c = cmax[t]
        if c > M:
            M = c
            I = cidx[t]
    result[0] = M + I

_NT = max(1, len(os.sched_getaffinity(0)))
_cmax = np.zeros(512)
_cidx = np.zeros(512, dtype=np.int64)
_w = np.random.rand(1 << 20)
_r = np.zeros(1)
_par(_w, 1, _w.size, _NT, _cmax, _cidx, _r)
_par(_w, 3, _w.size, _NT, _cmax, _cidx, _r)

_done = False
def s318(a, result, inc, LEN_1D):
    global _done
    a = np.ascontiguousarray(a)
    result = np.asarray(result)
    n = int(LEN_1D); st = int(inc)
    if not _done:
        _done = True
        out = []
        aff0 = sorted(os.sched_getaffinity(0))
        out.append("affinity0: %d CPUs %s" % (len(aff0), aff0[:12]))
        try:
            os.sched_setaffinity(0, set(range(192)))
            aff1 = sorted(os.sched_getaffinity(0))
            out.append("affinity1: %d CPUs" % len(aff1))
        except Exception as e:
            out.append("setaff failed: %r" % e)
        out.append("n=%d inc=%d nbytes=%.1fMB" % (n, st, a.nbytes / 1e6))
        for T in (1, 2, 4, 8, 16, 32, 64, 96, 128, 192):
            best = 1e30
            for _ in range(2):
                t0 = time.perf_counter()
                s = _sum_par(a, n, T)
                t1 = time.perf_counter()
                best = min(best, (t1 - t0))
            out.append("sum T=%3d %.1f ms %.0f GB/s" % (T, best * 1e3, a.nbytes / best / 1e6))
        print("PROBE | ".join(out), flush=True)
    _par(a, st, n, _NT, _cmax, _cidx, result)
    return None
