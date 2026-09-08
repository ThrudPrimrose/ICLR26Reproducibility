# TSVC tsvc_2 s152:  b = d*e ;  a += b*c   (in-place)
# Strategy: OpenMP C extension (compiled at import):
#   - s152_cauto : plain C, GCC auto-vectorized one-pass
#   - s152_avx   : explicit AVX-512, 16-wide, non-temporal stores (mode 1)
# Warm numba parallel fallback if the C build fails.
import ctypes
import os
import subprocess
import tempfile

_OMP_N = 12
if _OMP_N:
    os.environ["OMP_NUM_THREADS"] = str(_OMP_N)

import numpy as np
import numba
from numba import njit, prange

_C = r'''
#include <immintrin.h>
#include <stdint.h>

void s152_cauto(double *a, double *b, const double *c, const double *d, const double *e, int64_t n, int64_t chunk) {
#pragma omp parallel for schedule(dynamic, chunk)
  for (int64_t i = 0; i < n; ++i) {
    double t = d[i] * e[i];
    b[i] = t;
    a[i] += t * c[i];
  }
}

void s152_avx(double *a, double *b, const double *c, const double *d, const double *e,
              int64_t n, int64_t chunk, int64_t mode) {
  uintptr_t u = (uintptr_t)a | (uintptr_t)b | (uintptr_t)c | (uintptr_t)d | (uintptr_t)e;
  if ((u & 63) != 0) {
#pragma omp parallel for schedule(dynamic, chunk)
    for (int64_t i = 0; i < n; ++i) {
      double t = d[i] * e[i];
      b[i] = t;
      a[i] += t * c[i];
    }
    return;
  }
#pragma omp parallel for schedule(dynamic, chunk)
  for (int64_t i0 = 0; i0 < n; i0 += 16) {
    int64_t rem = n - i0;
    if (rem < 16) {
      for (int64_t i = i0; i < n; ++i) {
        double t = d[i] * e[i];
        b[i] = t;
        a[i] += t * c[i];
      }
      continue;
    }
    if (mode == 1) {
      {
        __m512d dd = _mm512_load_pd(d + i0);
        __m512d ee = _mm512_load_pd(e + i0);
        __m512d bb = _mm512_mul_pd(dd, ee);
        _mm512_stream_pd(b + i0, bb);
        __m512d cc = _mm512_load_pd(c + i0);
        __m512d aa = _mm512_load_pd(a + i0);
        aa = _mm512_add_pd(aa, _mm512_mul_pd(bb, cc));
        _mm512_stream_pd(a + i0, aa);
        dd = _mm512_load_pd(d + i0 + 8);
        ee = _mm512_load_pd(e + i0 + 8);
        bb = _mm512_mul_pd(dd, ee);
        _mm512_stream_pd(b + i0 + 8, bb);
        cc = _mm512_load_pd(c + i0 + 8);
        aa = _mm512_load_pd(a + i0 + 8);
        aa = _mm512_add_pd(aa, _mm512_mul_pd(bb, cc));
        _mm512_stream_pd(a + i0 + 8, aa);
      }
    } else {
      __m512d dd = _mm512_load_pd(d + i0);
      __m512d ee = _mm512_load_pd(e + i0);
      __m512d bb = _mm512_mul_pd(dd, ee);
      _mm512_store_pd(b + i0, bb);
      __m512d cc = _mm512_load_pd(c + i0);
      __m512d aa = _mm512_load_pd(a + i0);
      aa = _mm512_add_pd(aa, _mm512_mul_pd(bb, cc));
      _mm512_store_pd(a + i0, aa);
      dd = _mm512_load_pd(d + i0 + 8);
      ee = _mm512_load_pd(e + i0 + 8);
      bb = _mm512_mul_pd(dd, ee);
      _mm512_store_pd(b + i0 + 8, bb);
      cc = _mm512_load_pd(c + i0 + 8);
      aa = _mm512_load_pd(a + i0 + 8);
      aa = _mm512_add_pd(aa, _mm512_mul_pd(bb, cc));
      _mm512_store_pd(a + i0 + 8, aa);
    }
  }
}
'''

def _build():
    d = tempfile.mkdtemp(prefix="s152c_")
    src = os.path.join(d, "k.c")
    so = os.path.join(d, "k.so")
    with open(src, "w") as f:
        f.write(_C)
    r = subprocess.run(["gcc", "-O3", "-march=native", "-ffp-contract=off",
                        "-fopenmp", "-shared", "-fPIC", "-o", so, src],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    lib = ctypes.CDLL(so)
    fns = {}
    for name in ("s152_cauto", "s152_avx"):
        fn = getattr(lib, name)
        if name == "s152_avx":
            fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 3
        else:
            fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 2
        fn.restype = None
        fns[name] = fn
    return fns

try:
    _C_FNS = _build()
    _USE_C = True
except Exception:
    _C_FNS = None
    _USE_C = False

@njit(parallel=True)
def _fused_par(a, b, c, d, e, n):
    for i in prange(n):
        b[i] = d[i] * e[i]
    for i in prange(n):
        a[i] += b[i] * c[i]

@njit()
def _fused_ser(a, b, c, d, e, n):
    for i in range(n):
        b[i] = d[i] * e[i]
    for i in range(n):
        a[i] += b[i] * c[i]

_TINY = 65536
_MODE = "s152_avx"
_CHUNK = 131072

def _call_c(name, a, b, c, d, e, n):
    if name == "s152_avx":
        _C_FNS[name](a.ctypes.data, b.ctypes.data, c.ctypes.data,
                     d.ctypes.data, e.ctypes.data,
                     ctypes.c_int64(n), ctypes.c_int64(_CHUNK), ctypes.c_int64(1))
    else:
        _C_FNS[name](a.ctypes.data, b.ctypes.data, c.ctypes.data,
                     d.ctypes.data, e.ctypes.data,
                     ctypes.c_int64(n), ctypes.c_int64(_CHUNK))

def s152(a, b, c, d, e, LEN_1D):
    n = int(LEN_1D)
    if n == 0:
        return None
    if _USE_C and n > _TINY:
        _call_c(_MODE, a, b, c, d, e, n)
    elif n <= _TINY:
        _fused_ser(a, b, c, d, e, n)
    else:
        _fused_par(a, b, c, d, e, n)
    return None

def _warm():
    n = 1 << 20
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    d = np.zeros(n); e = np.zeros(n)
    _fused_ser(a, b, c, d, e, n)
    _fused_par(a, b, c, d, e, n)
    if _USE_C:
        for name in _C_FNS:
            _call_c(name, a, b, c, d, e, n)

_warm()
