"""probe"""
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np
import numba
from numba import prange

_C_SRC = r"""

#include <stdint.h>
#include <omp.h>
void k_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t n) {
    if (n <= 0) return;
    a[0] = b[0] * c[0];
    
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n; ++i) {
        a[i] = b[i] * c[i] + b[i-1] * c[i-1];
    }
}

"""

_C_FN = None
_EXTRA = ['-funroll-loops']


def _c_ok(arr):
    return arr.dtype == np.float64 and arr.ndim == 1 and arr.flags["C_CONTIGUOUS"]


def _build_c():
    global _C_FN
    d = tempfile.mkdtemp(prefix="s252k_")
    csrc = os.path.join(d, "k.c")
    lib = os.path.join(d, "libk.so")
    with open(csrc, "w") as f:
        f.write(_C_SRC)
    for cc in ("cc", "gcc", "clang"):
        try:
            r = subprocess.run(
                [cc, "-O3", "-march=native", "-fopenmp", "-ffp-contract=off"]
                + _EXTRA + ["-shared", "-fPIC", "-o", lib, csrc],
                capture_output=True, timeout=300)
            if r.returncode == 0 and os.path.exists(lib):
                break
        except Exception:
            pass
    try:
        dl = ctypes.CDLL(lib)
        fn = dl.k_fp64
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64]
        fn.restype = None
        

        _C_FN = (fn, lib)
    except Exception:
        _C_FN = None


_build_c()


@numba.njit(parallel=True)
def _s252(a, b, c):
    n = a.shape[0]
    if n > 0:
        a[0] = b[0] * c[0]
    for i in prange(1, n):
        a[i] = b[i] * c[i] + b[i - 1] * c[i - 1]


def s252(a, b, c, LEN_1D):
    n = a.shape[0]
    if n == 0:
        return None
    if _C_FN is not None and _c_ok(a) and _c_ok(b) and _c_ok(c):
        try:
            _C_FN[0](a.ctypes.data, b.ctypes.data, c.ctypes.data, n)
            return None
        except Exception:
            pass
    _s252(a, b, c)
    return None


def _warm():
    d64 = np.arange(32, dtype=np.float64)
    _s252(d64, d64, d64)


_warm()
