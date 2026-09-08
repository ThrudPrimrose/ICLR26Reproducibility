"""wf_diff_skew: difference-diagonal wavefront, a[i,j] += a[i-1,j] + a[i-1,j+1].

CPU implementation: a persistent OpenMP team with static j-stripes and a per-row
barrier, compiled to a shared object at import time (not timed). Falls back to a
numba serial kernel if the C toolchain is unavailable.
"""
import os
import subprocess
import sys
import tempfile
import ctypes

import numpy as np
import numba as nb

_C_SRC = r"""
#include <stdint.h>
#include <omp.h>

static inline void row_block(double* __restrict__ cur, const double* __restrict__ up,
                             int64_t j, int64_t j1) {
    int64_t k = j;
    for (; k + 8 <= j1; k += 8) {
        cur[k]   = cur[k]   + up[k]   + up[k+1];
        cur[k+1] = cur[k+1] + up[k+1] + up[k+2];
        cur[k+2] = cur[k+2] + up[k+2] + up[k+3];
        cur[k+3] = cur[k+3] + up[k+3] + up[k+4];
        cur[k+4] = cur[k+4] + up[k+4] + up[k+5];
        cur[k+5] = cur[k+5] + up[k+5] + up[k+6];
        cur[k+6] = cur[k+6] + up[k+6] + up[k+7];
        cur[k+7] = cur[k+7] + up[k+7] + up[k+8];
    }
    for (; k < j1; k++) cur[k] = cur[k] + up[k] + up[k+1];
}

void wf_diff_skew_c(double* __restrict__ a, const int64_t N, const int64_t T) {
    if (T <= 1) {
        for (int64_t i = 1; i < N; i++) {
            row_block(a + i*N, a + (i-1)*N, 0, N-1);
        }
        return;
    }
    #pragma omp parallel num_threads(T)
    {
        const int64_t t  = omp_get_thread_num();
        const int64_t Tt = omp_get_num_threads();
        const int64_t j0 = (N-1) * t / Tt;
        const int64_t j1 = (N-1) * (t+1) / Tt;
        for (int64_t i = 1; i < N; i++) {
            row_block(a + i*N, a + (i-1)*N, j0, j1);
            #pragma omp barrier
        }
    }
}
"""

_lib = None
_T_CORES = max(1, len(os.sched_getaffinity(0)))


def _build_lib():
    global _lib
    try:
        cdir = tempfile.mkdtemp(prefix="wfdskew_")
        csrc = os.path.join(cdir, "wf_diff_skew.c")
        so = os.path.join(cdir, "libwf_diff_skew.so")
        with open(csrc, "w") as f:
            f.write(_C_SRC)
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared",
             "-o", so, csrc],
            check=True, capture_output=True, timeout=120,
        )
        lib = ctypes.CDLL(so)
        lib.wf_diff_skew_c.argtypes = [ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int64, ctypes.c_int64]
        lib.wf_diff_skew_c.restype = None
        # keep the object alive for the process lifetime
        _build_lib._refs.append(so)
        return lib
    except Exception:
        return None


_build_lib._refs = []


def _get_lib():
    global _lib
    if _lib is None:
        _lib = _build_lib()
    return _lib


@nb.njit(fastmath=True)
def _nb_serial(a, N):
    for i in range(1, N):
        for j in range(N - 1):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i - 1, j + 1]


def wf_diff_skew(a, LEN_2D):
    N = int(LEN_2D)
    if N <= 1:
        return None
    if not (a.flags.c_contiguous and a.dtype == np.float64 and a.ndim == 2):
        _nb_serial(a, N)
        return None
    lib = _get_lib()
    if lib is not None:
        T = min(_T_CORES, 4)
        p = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.wf_diff_skew_c(p, N, T)
        return None
    _nb_serial(a, N)
    return None


# ---- import-time warmup (not charged) ----
try:
    wf_diff_skew(np.zeros((64, 64)), 64)
    _nb_serial(np.zeros((8, 8)), 8)
except Exception:
    pass
