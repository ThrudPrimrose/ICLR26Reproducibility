"""TSVC tsvc_2 s1244 -- optimized.

d[i] = new_a[i] + old_a[i+1],  a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]
(i in 0..LEN-2).  The a-update is a pure elementwise function of (b, c), so
the whole kernel is parallel-safe once the stale a[i+1] reads are ordered:
one OpenMP pass over contiguous per-thread chunks; each thread caches the
old value at its chunk start before a barrier, then streams b, c, a (shifted
read) and writes d, a.  48 bytes/elem traffic -- the race-free minimum.

Bit-exact to the reference op order ((b + c*c) + b*b) + c, no FMA contraction.
"""
import os
import sys
import ctypes
import shutil
import subprocess
import tempfile

import numpy as np
import numba as nb
from numba.np.ufunc import parallel as npar

def _s1244(a, b, c, d, LEN_1D):
    for i in nb.prange(LEN_1D - 1):
        d[i] = (b[i] + c[i]*c[i] + b[i]*b[i] + c[i]) + a[i+1]
    for i in nb.prange(LEN_1D - 1):
        a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]


_numba_k = nb.njit(parallel=True)(_s1244)

_C_SRC = r"""
#include <stdint.h>
#include <omp.h>

void k_single(double *a, const double *b, const double *c, double *d,
              int64_t n2, int nt) {
    if (n2 <= 0) return;
    if (nt < 1) nt = 1;
    if (nt > 63) nt = 63;
    double saved[64];
    #pragma omp parallel num_threads(nt)
    {
        int64_t tid = omp_get_thread_num();
        int64_t ntl = omp_get_num_threads();
        int64_t chunk = (n2 + ntl - 1) / ntl;
        int64_t lo = tid * chunk;
        int64_t hi = lo + chunk;
        if (hi > n2) hi = n2;
        saved[tid] = a[lo <= n2 ? lo : n2];
        if (tid == ntl - 1) saved[ntl] = a[n2];
        #pragma omp barrier
        for (int64_t i = lo; i + 1 < hi; i++) {
            double bi = b[i];
            double ci = c[i];
            double f = ((bi + ci*ci) + bi*bi) + ci;
            d[i] = f + a[i+1];
            a[i] = f;
        }
        if (hi > lo) {
            int64_t i = hi - 1;
            double bi = b[i];
            double ci = c[i];
            double f = ((bi + ci*ci) + bi*bi) + ci;
            d[i] = f + saved[tid+1];
            a[i] = f;
        }
    }
}
"""

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREBUILT_CANDIDATES = [
    os.path.join(_HERE, 'libtsvc_2_s1244.so'),
    '/shared/agent-14/libtsvc_2_s1244.so',
]


def _load_lib():
    for p in _PREBUILT_CANDIDATES:
        try:
            if os.path.exists(p):
                lib = ctypes.CDLL(p)
                _finish(lib.k_single)
                return
        except Exception:
            pass
    # Compile on the fly (import time, untimed).
    gcc = shutil.which('gcc') or shutil.which('cc')
    if not gcc:
        return
    try:
        td = tempfile.mkdtemp(prefix='tsvc_')
        csrc = os.path.join(td, 'k.c')
        so = os.path.join(td, 'libk.so')
        with open(csrc, 'w') as fh:
            fh.write(_C_SRC)
        subprocess.run(
            [gcc, '-O3', '-march=native', '-fopenmp', '-ffp-contract=off',
             '-shared', '-fPIC', '-o', so, csrc],
            check=True, capture_output=True, timeout=300)
        lib = ctypes.CDLL(so)
        _finish(lib.k_single)
    except Exception:
        pass


def _finish(fn):
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
    fn.restype = None
    global _cfn
    _cfn = fn


_cfn = None
try:
    _load_lib()
except Exception:
    _cfn = None

try:
    _NT = len(os.sched_getaffinity(0))
except Exception:
    _NT = 24
_NT = max(1, min(_NT, 63))

try:
    npar.set_num_threads(_NT)
except Exception:
    pass


def s1244(a, b, c, d, LEN_1D):
    if (_cfn is not None
            and a.dtype == np.float64 and b.dtype == np.float64
            and c.dtype == np.float64 and d.dtype == np.float64
            and a.flags['C_CONTIGUOUS'] and b.flags['C_CONTIGUOUS']
            and c.flags['C_CONTIGUOUS'] and d.flags['C_CONTIGUOUS']):
        _cfn(a.ctypes.data, b.ctypes.data, c.ctypes.data, d.ctypes.data,
             int(LEN_1D) - 1, _NT)
    else:
        _numba_k(a, b, c, d, LEN_1D)


# Warm everything at import time (untimed).
_t = 8
z = np.zeros(_t)
s1244(z.copy(), z.copy(), z.copy(), np.zeros(_t), _t)
_numba_k(z.copy(), z.copy(), z.copy(), np.zeros(_t), _t)
