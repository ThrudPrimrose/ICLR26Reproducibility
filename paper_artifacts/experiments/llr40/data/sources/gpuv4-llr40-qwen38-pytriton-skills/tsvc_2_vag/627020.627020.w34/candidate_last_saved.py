import numpy as np, numba, time
from numba import prange

@numba.njit(parallel=True)
def _fwd(a, b, ip, n):
    for i in prange(n):
        a[i] = b[ip[i]]
@numba.njit(parallel=True)
def _rev(a, b, ip, n):
    for k in prange(n):
        i = n - 1 - k
        a[i] = b[ip[i]]

def _warm():
    z = np.zeros(4096, dtype=np.float64)
    ip = np.arange(4096, dtype=np.int32)
    _fwd(z, z, ip, 4096); _rev(z, z, ip, 4096)
_warm()

def _t(f, a, b, ip, n):
    t0 = time.perf_counter(); f(a, b, ip, n); return 1e3*(time.perf_counter()-t0)

def vag(a, b, ip, LEN_1D):
    n = int(LEN_1D)
    r1 = _t(_rev, a, b, ip, n)
    f1 = _t(_fwd, a, b, ip, n)
    r2 = _t(_rev, a, b, ip, n)
    f2 = _t(_fwd, a, b, ip, n)
    print(f"DBG n={n} rev1={r1:.2f} fwd1={f1:.2f} rev2={r2:.2f} fwd2={f2:.2f}", flush=True)
    _fwd(a, b, ip, n)  # leave correct result
    return None
