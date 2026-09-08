"""Optimized implementation of fuse_move_ifs (TSVC tsvc_2_5).

Reference:
  for i: if cond[i] > 0: a[i,:] = src[i,:] * 2
  if K > 0: b = src + 1   (elementwise, full)

Strategy: the serial numba baseline streams ~1.9 GB (fuzzed size), so the
win is multithreaded streaming. A small C+OpenMP kernel is compiled at
import time (untimed) and called through ctypes on the in-place buffers.
Fallbacks: numba (parallel) -> numpy, in case the judge node lacks a
compiler.
"""
import ctypes
import os
import shutil
import subprocess
import tempfile

import numpy as np

_C_SRC = r"""
#include <stdint.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b,
                        const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int64_t i = 0; i < N; ++i) {
    const double *sr = src + i * N;
    if (cond[i] > 0.0) {
      double *ar = a + i * N;
      for (int64_t j = 0; j < N; ++j) ar[j] = sr[j] * 2.0;
    }
    if (K > 0) {
      double *br = b + i * N;
      for (int64_t j = 0; j < N; ++j) br[j] = sr[j] + 1.0;
    }
  }
}
"""


def _build_omp():
    """Compile the C kernel at import time; return (lib, n_threads_hint) or None."""
    if shutil.which("gcc") is None:
        return None
    try:
        d = tempfile.mkdtemp(prefix="fmif_")
        cpath = os.path.join(d, "k.c")
        lpath = os.path.join(d, "libk.so")
        with open(cpath, "w") as f:
            f.write(_C_SRC)
        r = subprocess.run(
            ["gcc", "-O3", "-fopenmp", "-fPIC", "-shared", "-o", lpath, cpath],
            capture_output=True, timeout=300,
        )
        if r.returncode != 0:
            return None
        lib = ctypes.CDLL(lpath)
        fn = lib.fuse_move_ifs_fp64
        fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64, ctypes.c_int64]
        fn.restype = None
        return (lib, fn)
    except Exception:
        return None


_OMP = _build_omp()

# ---- fallbacks (also built lazily so import stays fast) ----
def _numpy_fallback(a, b, src, cond, LEN_2D, K):
    m = cond[:LEN_2D] > 0.0
    if m.any():
        a[m] = src[m] * 2.0
    if K > 0:
        np.add(src, 1.0, out=b)


def _numba_fallback(a, b, src, cond, LEN_2D, K):
    global _NB
    try:
        _nb = _NB
    except NameError:
        _nb = None
    if _nb is None:
        import numba as nb
        from numba import prange

        @nb.njit(parallel=True, fastmath=True)
        def _k(a, b, src, cond, K, N):
            for i in prange(N):
                s = src[i]
                if cond[i] > 0.0:
                    t = a[i]
                    for j in range(N):
                        t[j] = s[j] * 2.0
                if K > 0:
                    u = b[i]
                    for j in range(N):
                        u[j] = s[j] + 1.0
        _nb = _k
        _NB = _nb
    _nb(a, b, src, cond, K, LEN_2D)


def _contig64(x):
    if x.dtype != np.float64 or not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x, dtype=np.float64)
    return x


def fuse_move_ifs(a, b, src, cond, LEN_2D, K):
    N = int(LEN_2D)
    if _OMP is not None:
        a2 = _contig64(a)
        b2 = _contig64(b)
        s2 = _contig64(src)
        c2 = _contig64(cond)
        _OMP[1](a2.ctypes.data, b2.ctypes.data, c2.ctypes.data, s2.ctypes.data,
                int(K), N)
        if a2 is not a:
            a[...] = a2
        if b2 is not b:
            b[...] = b2
        return
    try:
        _numba_fallback(a, b, src, cond, N, K)
    except Exception:
        _numpy_fallback(a, b, src, cond, N, K)
