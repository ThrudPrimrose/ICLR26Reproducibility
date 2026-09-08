"""Optimized s275 (TSVC 2.0 scan kernel).

Each column i is an independent recurrence  aa[j,i] = aa[j-1,i] + bb[j,i]*cc[j,i]
(gated by aa[0,i] > 0).  Columns are processed in contiguous blocks so that every
iteration of the block streams row-contiguous memory; blocks run in parallel.
A C extension (built at import time, untimed) gives AVX-512 masked-vector code;
a numba variant is kept as a fallback if no compiler is available.
"""
import ctypes
import os
import subprocess
import tempfile

import numpy as np
import numba
from numba import njit, prange

_CSRC = r"""
#include <stdint.h>
#include <omp.h>

void tsvc2_s275(const double *restrict a0, double *restrict aa,
                const double *restrict bb, const double *restrict cc,
                const int64_t N, const int64_t B) {
  #pragma omp parallel for schedule(dynamic, 1)
  for (int64_t i0 = 0; i0 < N; i0 += B) {
    int64_t i1 = i0 + B;
    if (i1 > N) i1 = N;
    int64_t len = i1 - i0;
    for (int64_t j = 1; j < N; j++) {
      const double *ap = aa + (j - 1) * N + i0;
      double *ad = aa + (int64_t)j * N + i0;
      const double *bp = bb + (int64_t)j * N + i0;
      const double *cp = cc + (int64_t)j * N + i0;
      const double *m = a0 + i0;
      for (int64_t i = 0; i < len; i++)
        if (m[i] > 0.0)
          ad[i] = ap[i] + bp[i] * cp[i];
    }
  }
}

int64_t s275_max_threads(void) {
  return (int64_t)omp_get_max_threads();
}
"""

_B = 512  # columns per chunk


def _build_c():
    d = tempfile.mkdtemp(prefix="s275ext_")
    csrc = os.path.join(d, "kern.c")
    so = os.path.join(d, "libkern.so")
    with open(csrc, "w") as f:
        f.write(_CSRC)
    subprocess.run(
        ["gcc", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared",
         "-o", so, csrc],
        check=True, capture_output=True, timeout=120)
    lib = ctypes.CDLL(so)
    lib.tsvc2_s275.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64,
                                                       ctypes.c_int64]
    lib.tsvc2_s275.restype = None
    lib.s275_max_threads.argtypes = []
    lib.s275_max_threads.restype = ctypes.c_int64
    return lib


_LIB = None
try:
    _LIB = _build_c()
    _NT = max(1, int(numba.get_num_threads()))
    try:
        _gomp = ctypes.CDLL("libgomp.so.1")
        _gomp.omp_set_num_threads.restype = None
        _gomp.omp_set_num_threads.argtypes = [ctypes.c_int]
        _gomp.omp_set_num_threads(_NT)
    except Exception:
        pass
    _d0 = np.ones((8, 8))
    _LIB.tsvc2_s275(_d0.ctypes.data, _d0.ctypes.data, _d0.ctypes.data,
                    _d0.ctypes.data, 8, 8)  # create GOMP pool (untimed)
    del _d0
except Exception:
    _LIB = None


@njit(parallel=True, fastmath=True, cache=False)
def _scan_blocks(aa, bb, cc, N, T):
    B = (N + T - 1) // T
    for t in prange(T):
        i0 = t * B
        i1 = i0 + B
        if i1 > N:
            i1 = N
        for j in range(1, N):
            for i in range(i0, i1):
                if aa[0, i] > 0.0:
                    aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]


def _warm():
    d = np.ones((8, 8))
    _scan_blocks(d, d.copy(), d.copy(), 8, max(1, numba.get_num_threads()))


_warm()
del _warm

def s275(aa, bb, cc, LEN_2D):
    N = int(LEN_2D)
    if _LIB is not None:
        _LIB.tsvc2_s275(aa.ctypes.data, aa.ctypes.data, bb.ctypes.data,
                        cc.ctypes.data, N, _B)
    else:
        _scan_blocks(aa, bb, cc, N, min(max(1, numba.get_num_threads()), N))
