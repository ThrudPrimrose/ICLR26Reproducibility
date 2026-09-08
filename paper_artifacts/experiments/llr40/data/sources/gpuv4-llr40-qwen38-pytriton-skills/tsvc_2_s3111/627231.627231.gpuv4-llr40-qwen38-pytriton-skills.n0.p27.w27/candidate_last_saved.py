import os
_K = max(1, min(len(os.sched_getaffinity(0)), 64))
os.environ.setdefault("NUMBA_NUM_THREADS", str(_K))
import numpy as np
import numba as nb
from numba import prange

try:
    nb.set_num_threads(_K)
except Exception:
    pass

@nb.njit(fastmath=True)
def _serial(a, n):
    s = 0.0
    for i in range(n):
        if a[i] > 0.0:
            s += a[i]
    return s

@nb.njit(fastmath=True, parallel=True)
def _par(a, n):
    s = 0.0
    for i in prange(n):
        if a[i] > 0.0:
            s += a[i]
    return s

@nb.njit(fastmath=True, parallel=True)
def _par_pure(a, n):
    s = 0.0
    for i in prange(n):
        s += a[i]
    return s

_PAR_MIN = 1 << 18

_diag = False

def s3111(a, b, LEN_1D):
    global _diag
    n = int(LEN_1D)
    a = np.ascontiguousarray(a, dtype=np.float64)
    if not _diag and n > (1 << 20):
        _diag = True
        import time
        t0 = time.perf_counter(); r1 = _par(a, n); t1 = time.perf_counter()
        r2 = _par_pure(a, n); t2 = time.perf_counter()
        r3 = _par(a, n); t3 = time.perf_counter()
        s = _serial(a, n); t4 = time.perf_counter()
        print(f"DIAG n={n} aff={len(os.sched_getaffinity(0))}")
        print(f"DIAG masked par: {(t1-t0)*1e3:.2f} ms {(t3-t2)*1e3:.2f} ms  -> {8*n/(t3-t2)/1e9:.0f} GB/s")
        print(f"DIAG pure   par: {(t2-t1)*1e3:.2f} ms  -> {8*n/(t2-t1)/1e9:.0f} GB/s")
        print(f"DIAG serial   : {(t4-t3)*1e3:.2f} ms  -> {8*n/(t4-t3)/1e9:.0f} GB/s")
        print(f"DIAG check masked==serial: {r1==s}  {r1} vs {s}")
    r = _par(a, n) if n >= _PAR_MIN else _serial(a, n)
    b[0] = r
    return None

_w = np.zeros(1 << 17)
_wb = np.zeros(2)
s3111(_w, _wb, _w.shape[0])
_w2 = np.random.rand(1 << 21)
s3111(_w2, _wb, _w2.shape[0])
