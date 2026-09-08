import os
import subprocess
import sys
import tempfile
import ctypes

import numpy as np
import numba as nb

_C_SRC = r"""
#include <immintrin.h>
#include <stdint.h>

static inline void row_b(double *restrict bi, const double *restrict si, int64_t n) {
  int64_t j = 0;
  uintptr_t mis = (uintptr_t)bi & 31;
  if (mis) {
    int64_t k = (32 - mis) >> 3;
    if (k > n) k = n;
    while (j < k) bi[j] = si[j] + 1.0, ++j;
  }
  for (; j + 4 <= n; j += 4)
    _mm256_stream_pd(bi + j, _mm256_add_pd(_mm256_loadu_pd(si + j), _mm256_set1_pd(1.0)));
  for (; j < n; ++j) bi[j] = si[j] + 1.0;
}

static inline void row_a(double *restrict ai, const double *restrict si, int64_t n) {
  int64_t j = 0;
  uintptr_t mis = (uintptr_t)ai & 31;
  if (mis) {
    int64_t k = (32 - mis) >> 3;
    if (k > n) k = n;
    while (j < k) ai[j] = si[j] * 2.0, ++j;
  }
  for (; j + 4 <= n; j += 4)
    _mm256_stream_pd(ai + j, _mm256_mul_pd(_mm256_loadu_pd(si + j), _mm256_set1_pd(2.0)));
  for (; j < n; ++j) ai[j] = si[j] * 2.0;
}

void fuse_move_ifs_fp64(double *restrict a, double *restrict b,
                        const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  if (K > 0) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
      double *bi = b + i * LEN_2D;
      const double *si = src + i * LEN_2D;
      if (cond[i] > 0.0) {
        row_a(a + i * LEN_2D, si, LEN_2D);
      }
      row_b(bi, si, LEN_2D);
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
      if (cond[i] > 0.0) row_a(a + i * LEN_2D, src + i * LEN_2D, LEN_2D);
    }
  }
}
"""


def _which(name):
    for p in os.environ.get("PATH", "").split(os.pathsep):
        fp = os.path.join(p, name)
        if os.path.isfile(fp) and os.access(fp, os.X_OK):
            return fp
    return None


def _build_lib():
    for cc in ("gcc", "cc"):
        exe = _which(cc)
        if not exe:
            continue
        d = tempfile.mkdtemp(prefix="fmi_")
        cpath = os.path.join(d, "fmi.c")
        so = os.path.join(d, "libfmi.so")
        try:
            with open(cpath, "w") as f:
                f.write(_C_SRC)
            subprocess.run([exe, "-O3", "-march=native", "-fopenmp", "-shared",
                            "-fPIC", "-o", so, cpath], check=True,
                           capture_output=True, timeout=180)
            lib = ctypes.CDLL(so)
            lib.fuse_move_ifs_fp64.restype = None
            lib.fuse_move_ifs_fp64.argtypes = [ctypes.c_void_p] * 4 + \
                [ctypes.c_int64, ctypes.c_int64]
            return lib, so
        except Exception:
            pass
    return None, None


_LIB, _SO = _build_lib()
if _LIB is not None:
    _KEEP = open(_SO, "rb")


@nb.njit(parallel=True)
def _fallback(a, b, src, cond, LEN_2D, K):
    for i in nb.prange(LEN_2D):
        if cond[i] > 0.0:
            for j in range(LEN_2D):
                a[i, j] = src[i, j] * 2.0
        if K > 0:
            for j in range(LEN_2D):
                b[i, j] = src[i, j] + 1.0


def _warm():
    global _LIB
    m = 16
    aa = np.zeros((m, m)); bb = np.zeros((m, m)); ss = np.ones((m, m))
    cc = np.array([1.0, -1.0] * (m // 2))
    _fallback(aa, bb, ss, cc, m, 1)
    _fallback(aa, bb, ss, cc, m, -1)
    if _LIB is not None:
        try:
            aa2 = np.zeros((m, m)); bb2 = np.zeros((m, m))
            _LIB.fuse_move_ifs_fp64(aa2.ctypes.data, bb2.ctypes.data, cc.ctypes.data,
                                    ss.ctypes.data, 1, m)
            mask = cc > 0.0
            exp_a = np.where(mask[:, None], ss * 2.0, np.zeros_like(ss))
            ok = np.array_equal(aa2, exp_a) and np.array_equal(bb2, ss + 1.0)
            aa3 = np.zeros((m, m)); bb3 = np.ones((m, m))
            _LIB.fuse_move_ifs_fp64(aa3.ctypes.data, bb3.ctypes.data, cc.ctypes.data,
                                    ss.ctypes.data, -1, m)
            ok = ok and np.array_equal(aa3, exp_a) and np.array_equal(bb3, np.ones((m, m)))
            if not ok:
                raise RuntimeError("C self-verify failed")
        except Exception:
            _LIB = None


_warm()


def fuse_move_ifs(a, b, src, cond, LEN_2D, K):
    if (_LIB is not None and a.dtype == np.float64 and b.dtype == np.float64
            and src.dtype == np.float64 and cond.dtype == np.float64
            and a.flags["C_CONTIGUOUS"] and b.flags["C_CONTIGUOUS"]
            and src.flags["C_CONTIGUOUS"] and cond.flags["C_CONTIGUOUS"]
            and a.shape == (LEN_2D, LEN_2D) and b.shape == (LEN_2D, LEN_2D)
            and src.shape == (LEN_2D, LEN_2D) and cond.shape == (LEN_2D,)):
        _LIB.fuse_move_ifs_fp64(a.ctypes.data, b.ctypes.data,
                                cond.ctypes.data, src.ctypes.data, K, LEN_2D)
        return None
    _fallback(a, b, src, cond, LEN_2D, K)
    return None
