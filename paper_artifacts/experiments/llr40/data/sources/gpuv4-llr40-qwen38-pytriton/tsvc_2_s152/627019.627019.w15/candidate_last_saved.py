"""TSVC tsvc_2 s152 optimized (python arm).

Reference semantics (numpy oracle):
    for i in range(LEN_1D): b[i] = d[i] * e[i]
    for i in range(LEN_1D): a[i] = a[i] + b[i] * c[i]

Final: b[i] = d[i]*e[i];  a[i] += (d[i]*e[i])*c[i].  The two loops only share
state per-index, so one fused pass (6 memory streams instead of 7) gives the
identical rounded result.  The hot path is a small C kernel compiled at import
time (not timed) with AVX-512 + OpenMP; each iteration is independent, so the
parallel result is bit-identical to the serial reference (built with
-ffp-contract=off, no unsafe math).

Fallbacks: single-CPU -> warmed numba njit fused loop; unusual dtypes or
strided input -> plain numpy (same expression shape as the oracle).
"""
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np
import numba

_C_SRC = r"""
#include <stdint.h>
#include <omp.h>
void tsvc_2_s152_k(double* a, double* b, const double* c, const double* d, const double* e, int64_t n) {
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    double t = d[i] * e[i];
    b[i] = t;
    a[i] += t * c[i];
  }
}
"""


@numba.njit
def _s152_fused(a, b, c, d, e, n):
    for i in range(n):
        t = d[i] * e[i]
        b[i] = t
        a[i] += t * c[i]


# Warm the numba JIT at import time (import is not timed; first call is).
_w = np.ones(64)
_s152_fused(_w, _w.copy(), _w.copy(), _w.copy(), _w.copy(), 64)
del _w

_lib = None
_omp_set_num_threads = None
_f = None
_MAX_T = 8


def _build_c():
    """Compile + load the C/OMP kernel at import time (outside the timer)."""
    global _lib, _omp_set_num_threads, _f
    tmp = tempfile.mkdtemp(prefix="tsvc152_")
    csrc = os.path.join(tmp, "tsvc_2_s152_k.c")
    so = os.path.join(tmp, "libtsvc_2_s152_k.so")
    try:
        with open(csrc, "w") as fh:
            fh.write(_C_SRC)
        r = subprocess.run(
            ["gcc", "-O3", "-march=native", "-fopenmp", "-ffp-contract=off",
             "-shared", "-fPIC", "-o", so, csrc],
            capture_output=True, timeout=300)
        if r.returncode != 0:
            return False
        lib = ctypes.CDLL(so)
        fn = lib.tsvc_2_s152_k
        fn.argtypes = [ctypes.POINTER(ctypes.c_double)] * 5 + [ctypes.c_int64]
        fn.restype = None
        set_nt = lib.omp_set_num_threads
        set_nt.argtypes = [ctypes.c_int]
        set_nt.restype = None
        _lib, _f, _omp_set_num_threads = lib, fn, set_nt
    except Exception:
        return False
    # Warm the OMP pool and the .so text pages before any timed call.
    try:
        w = np.ones(1 << 20)
        t = max(1, min(_MAX_T, len(os.sched_getaffinity(0))))
        _omp_set_num_threads(t)
        P = ctypes.POINTER(ctypes.c_double)
        _f(w.ctypes.data_as(P), w.ctypes.data_as(P), w.ctypes.data_as(P),
           w.ctypes.data_as(P), w.ctypes.data_as(P), w.size)
        del w
    except Exception:
        pass
    return True


try:
    _build_c()
except Exception:
    _lib = None


def _all_c_f64(a, b, c, d, e):
    return (
        isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
        and isinstance(c, np.ndarray) and isinstance(d, np.ndarray)
        and isinstance(e, np.ndarray)
        and a.dtype == np.float64 and b.dtype == np.float64
        and c.dtype == np.float64 and d.dtype == np.float64
        and e.dtype == np.float64
        and a.flags.c_contiguous and b.flags.c_contiguous
        and c.flags.c_contiguous and d.flags.c_contiguous
        and e.flags.c_contiguous
        and b.flags.writeable and a.flags.writeable
    )


def s152(a, b, c, d, e, LEN_1D):
    n = LEN_1D
    if n <= 0:
        return None
    if not _all_c_f64(a, b, c, d, e):
        b[:n] = d[:n] * e[:n]
        a[:n] += b[:n] * c[:n]
        return None
    if _f is None:
        _s152_fused(a, b, c, d, e, n)
        return None
    nt = len(os.sched_getaffinity(0))
    if nt > _MAX_T:
        nt = _MAX_T
    elif nt < 1:
        nt = 1
    if nt == 1:
        _s152_fused(a, b, c, d, e, n)
        return None
    _omp_set_num_threads(nt)
    P = ctypes.POINTER(ctypes.c_double)
    _f(a.ctypes.data_as(P), b.ctypes.data_as(P), c.ctypes.data_as(P),
       d.ctypes.data_as(P), e.ctypes.data_as(P), n)
    return None
