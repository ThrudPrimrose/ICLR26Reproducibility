"""Optimized vpvts: a[i] += b[i] * S (in-place; also returns a).

Primary path: prebuilt AVX-512 + OpenMP streaming kernel (ctypes).
Fallbacks: parallel numba (warmed at import), then vectorised NumPy.
"""
import os
import ctypes
import numpy as np

# --- load the prebuilt native kernel (needs only libgomp + libc) -------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None
for _p in (os.path.join(_HERE, "libvpvts_axpy.so"),
           "/shared/agent-35/libvpvts_axpy.so"):
    try:
        _LIB = ctypes.CDLL(_p)
        _LIB.vpvts_axpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int64, ctypes.c_double, ctypes.c_int]
        _LIB.vpvts_axpy.restype = None
        break
    except OSError:
        _LIB = None

# --- numba fallback (compiled + warmed at import time) -----------------------
_HAVE_NUMBA = False
_NTHREADS = 1
try:
    import numba as _nb
    from numba import prange

    try:
        _NTHREADS = len(os.sched_getaffinity(0))
    except Exception:
        _NTHREADS = os.cpu_count() or 1
    _NTHREADS = max(1, min(_NTHREADS, 32))
    _nb.set_num_threads(_NTHREADS)

    @_nb.njit(parallel=True, fastmath=True)
    def _par(a, b, n, S):
        for i in prange(n):
            a[i] = a[i] + b[i] * S

    @_nb.njit(fastmath=True)
    def _ser(a, b, n, S):
        for i in range(n):
            a[i] = a[i] + b[i] * S

    _n = 1 << 20
    _a = np.zeros(_n)
    _b = np.ones(_n)
    for _ in range(2):
        _par(_a, _b, _n, 3)
        _ser(_a, _b, _n, 3)
    _par(_a, _b, _n, np.int64(3))
    _ser(_a, _b, _n, np.int64(3))
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

_PAR_MIN = 1 << 22          # parallel numba only pays above this
_LIB_MT_MIN = 1 << 20       # use the threaded C path above this


def vpvts(a, b, LEN_1D, S):
    n = int(LEN_1D)
    if (a.dtype == np.float64 and b.dtype == np.float64
            and a.ndim == 1 and b.ndim == 1
            and a.flags["C_CONTIGUOUS"] and b.flags["C_CONTIGUOUS"]):
        if _LIB is not None:
            _LIB.vpvts_axpy(a.ctypes.data, b.ctypes.data, n, float(S),
                            _NTHREADS if n >= _LIB_MT_MIN else 1)
            return a
        if _HAVE_NUMBA:
            if n >= _PAR_MIN and _NTHREADS > 1:
                _par(a, b, n, S)
            else:
                _ser(a, b, n, S)
            return a
    # generic fallback: any dtype / layout
    a[:n] = a[:n] + b[:n] * S
    return a
