import os as _os, sys as _sys, time as _time, shutil as _shutil, subprocess as _sp, tempfile as _tf
import ctypes

def _p(*a):
    print(*a)
    _sys.stdout.flush()

_p(f"PROBE affinity={_os.sched_getaffinity(0)} cpu_count={_os.cpu_count()}")
for f in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
    try:
        _p(f"PROBE {f} = {open(f).read().strip()!r}")
    except OSError as ex:
        _p(f"PROBE {f} = MISSING {ex}")

CC = None
for cand in ("cc", "gcc", "clang"):
    if _shutil.which(cand):
        CC = cand
        break
_p(f"PROBE compiler = {CC}")

CSRC = r'''
#include <stdint.h>
#include <math.h>
#define RESTRICT __restrict
__attribute__((visibility("default")))
void tsvc_2_s319_fp64(double* RESTRICT a, double* RESTRICT b, const double* RESTRICT c,
                      const double* RESTRICT d, const double* RESTRICT e, const int64_t n) {
  double sum = 0.0;
  #pragma omp parallel for schedule(static, 4096) reduction(+:sum)
  for (int64_t i = 0; i < n; ++i) {
    double ai = c[i] + d[i];
    a[i] = ai;
    double bi = c[i] + e[i];
    b[i] = bi;
    sum += ai + bi;   /* NOTE: probe only, rounding differs from ref */
  }
  b[0] = sum;
  (void)0;
}
'''
_lib = None
if CC:
    try:
        d = _tf.mkdtemp()
        src = _os.path.join(d, "k.c")
        so = _os.path.join(d, "k.so")
        with open(src, "w") as f:
            f.write(CSRC)
        r = _sp.run([CC, "-O3", "-mavx2", "-mfma", "-fopenmp", "-shared", "-fPIC", src, "-o", so, "-lm"],
                    capture_output=True, text=True, timeout=120)
        _p(f"PROBE compile rc={r.returncode} {r.stderr.strip()[:200]!r}")
        if r.returncode == 0:
            _lib = ctypes.CDLL(so)
            _f = _lib.tsvc_2_s319_fp64
            _f.argtypes = [ctypes.POINTER(ctypes.c_double)] * 5 + [ctypes.c_int64]
            _f.restype = None
            _p("PROBE loaded C lib")
    except Exception as ex:
        _p(f"PROBE C-builtin failed: {type(ex).__name__} {ex}")

_os.environ.setdefault('NUMBA_NUM_THREADS','8')
import numpy as np
from numba import njit, prange

@njit
def _seq(a, b, c, d, e):
    n = c.shape[0]
    s = 0.0
    for i in range(n):
        ai = c[i] + d[i]; a[i] = ai; s = s + ai
        bi = c[i] + e[i]; b[i] = bi; s = s + bi
    b[0] = s

@njit(parallel=True)
def _par(a, b, c, d, e, nt):
    n = c.shape[0]
    p0 = np.empty(nt, dtype=np.float64)
    p1 = np.empty(nt, dtype=np.float64)
    for j in prange(nt):
        lo = (n * j) // nt
        hi = (n * (j + 1)) // nt
        m = (lo + hi) // 2
        s0 = 0.0; s1 = 0.0
        for i in range(lo, m):
            ai = c[i] + d[i]; a[i] = ai; s0 = s0 + ai
            bi = c[i] + e[i]; b[i] = bi; s1 = s1 + bi
        for i in range(m, hi):
            ai = c[i] + d[i]; a[i] = ai; s0 = s0 + ai
            bi = c[i] + e[i]; b[i] = bi; s1 = s1 + bi
        p0[j] = s0; p1[j] = s1
    s0 = 0.0; s1 = 0.0
    for j in range(nt):
        s0 = s0 + p0[j]; s1 = s1 + p1[j]
    b[0] = s0 + s1

_SMALL = 16384

def s319(a, b, c, d, e, LEN_1D):
    n = LEN_1D
    if n < _SMALL:
        _seq(a, b, c, d, e)
        return
    nt = min(16, max(1, n >> 14))
    _par(a, b, c, d, e, nt)
    if _lib is not None and n >= 1 << 20:
        P = ctypes.POINTER(ctypes.c_double)
        def timethread(fn, R=3):
            fn()
            t0 = _time.perf_counter()
            for _ in range(R): fn()
            return (_time.perf_counter() - t0) / R
                try:
            gomp = ctypes.CDLL("libgomp.so.1")
            set_nt = gomp.omp_set_num_threads
            set_nt.restype = ctypes.c_int
        except OSError:
            set_nt = None
        P = ctypes.POINTER(ctypes.c_double)
        ap, bp, cp, dp, ep = (x.ctypes.data_as(P) for x in (a, b, c, d, e))
        if set_nt is not None:
            set_nt(1)
            t1 = timethread(lambda: _f(ap, bp, cp, dp, ep, n))
            _p(f"PROBE C-1t   {t1*1e6:9.1f} us  {40*n/t1/1e9:8.1f} GB/s")
        for T in (4, 8, 16):
            if set_nt: set_nt(T)
            tT = timethread(lambda: _f(ap, bp, cp, dp, ep, n))
            _p(f"PROBE C-{T:2d}t {tT*1e6:9.1f} us  {40*n/tT/1e9:8.1f} GB/s")
        tn = timethread(lambda: _par(a, b, c, d, e, nt))
        _p(f"PROBE numba nt={nt} {tn*1e6:9.1f} us  {40*n/tn/1e9:8.1f} GB/s")
        # restore a correct-ish b[0] via numba path
        _par(a, b, c, d, e, nt)
    else:
        _par(a, b, c, d, e, nt)

def _warm():
    c = np.random.rand(2 * _SMALL + 1); d = np.random.rand(2 * _SMALL + 1); ee = np.random.rand(2 * _SMALL + 1)
    a = np.empty(2 * _SMALL + 1); b = np.empty(2 * _SMALL + 1)
    s319(a, b, c, d, ee, 1000)
    s319(a, b, c, d, ee, 2 * _SMALL + 1)

_warm()
