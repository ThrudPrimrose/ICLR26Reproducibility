import os, sys, time
import numpy as np
from numba import njit, prange, config

@njit(cache=False)
def _serial(a, b, ip, n):
    for i in range(n):
        a[i] = b[ip[i]]

@njit(parallel=True, cache=False)
def _par(a, b, ip, n):
    for i in prange(n):
        a[i] = b[ip[i]]

_w = np.empty(1024); _b = np.arange(1024.0); _ip = np.arange(1024, dtype=np.int32)
_serial(_w, _b, _ip, 1024); _par(_w, _b, _ip, 1024)
_serial(_w, _b, _ip.astype(np.int64), 1024); _par(_w, _b, _ip.astype(np.int64), 1024)

def vag(a, b, ip, LEN_1D):
    n = int(LEN_1D)
    p = lambda *x: print(*x, flush=True)
    p("N =", n, "bytes20N =", n*20)
    p("dtypes", a.dtype, b.dtype, ip.dtype, "shapes", a.shape, b.shape, ip.shape,
      "strides", a.strides, b.strides, ip.strides, "contig", a.flags.c_contiguous, b.flags.c_contiguous, ip.flags.c_contiguous)
    p("ip min/max", int(ip.min()), int(ip.max()))
    k = min(65536, n)
    s = ip[:k]
    p("sample: eq_i frac", float((s == np.arange(k)).mean()),
      "mean|d|", float(np.abs(s.astype(np.int64) - np.arange(k)).mean()),
      "std|d|", float(np.abs(s.astype(np.int64) - np.arange(k)).std()),
      "uniq in sample", len(np.unique(s)))
    p("cpu_count", os.cpu_count(), "affinity", len(os.sched_getaffinity(0)),
      "numba cfg threads", config.NUMBA_NUM_THREADS)
    env = {kk: os.environ.get(kk) for kk in ("OMP_NUM_THREADS","NUMBA_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_DYNAMIC","OMP_MAX_ACTIVE_LEVELS")}
    p("env", env)
    t0 = time.perf_counter(); _serial(a, b, ip, n); t1 = time.perf_counter()
    p("t_serial_ms", (t1-t0)*1e3, "ns/elem", (t1-t0)*1e9/n)
    t0 = time.perf_counter(); _par(a, b, ip, n); t1 = time.perf_counter()
    p("t_par_ms", (t1-t0)*1e3, "ns/elem", (t1-t0)*1e9/n)
    t0 = time.perf_counter(); a[...] = b[ip]; t1 = time.perf_counter()
    p("t_numpy_ms", (t1-t0)*1e3, "ns/elem", (t1-t0)*1e9/n)
    assert np.array_equal(a, b[ip])
    p("OK")
