import ctypes
import os

import numpy as np

_SO_PATH = "/shared/agent-23/libk2710.so"

_lib = None
_HAVE_SO = False

try:
    with open("/proc/cpuinfo") as _f:
        _has_avx512 = "avx512f" in _f.read()
except Exception:
    _has_avx512 = False

if _has_avx512 and os.path.exists(_SO_PATH):
    try:
        _lib = ctypes.CDLL(_SO_PATH)
        _lib.s2710_c.argtypes = [ctypes.POINTER(ctypes.c_double)] * 6 + [ctypes.c_longlong]
        _lib.s2710_c.restype = None
        _t = np.zeros(64)
        _tx = np.array([1.0])
        _p = _t.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _lib.s2710_c(_p, _p, _p, _p, _p,
                     _tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 64)
        _HAVE_SO = True
    except Exception:
        _lib = None

try:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True, cache=False)
    def _s2710_nb(a, b, c, d, e, x, LEN_1D):
        big = LEN_1D > 10
        pos = x[0] > 0.0
        for i in prange(LEN_1D):
            if a[i] > b[i]:
                a[i] = a[i] + b[i] * d[i]
                if big:
                    c[i] = c[i] + d[i] * d[i]
                else:
                    c[i] = d[i] * e[i] + 1.0
            else:
                b[i] = a[i] + e[i] * e[i]
                if pos:
                    c[i] = a[i] + d[i] * d[i]
                else:
                    c[i] = c[i] + e[i] * e[i]

    _w = np.zeros(256)
    _wx = np.zeros(1)
    _s2710_nb(_w.copy(), _w.copy(), _w.copy(), _w, _w.copy(), _wx, 256)
except Exception:
    _s2710_nb = None

_P64 = ctypes.POINTER(ctypes.c_double)
_F64 = np.dtype(np.float64)


def s2710(a, b, c, d, e, x, LEN_1D):
    if _HAVE_SO and a.dtype is _F64 and a.ndim == 1:
        if (b.dtype is _F64 and c.dtype is _F64 and d.dtype is _F64 and e.dtype is _F64
                and x.dtype is _F64
                and a.flags["C_CONTIGUOUS"] and b.flags["C_CONTIGUOUS"] and c.flags["C_CONTIGUOUS"]
                and d.flags["C_CONTIGUOUS"] and e.flags["C_CONTIGUOUS"] and x.flags["C_CONTIGUOUS"]):
            _lib.s2710_c(a.ctypes.data_as(_P64), b.ctypes.data_as(_P64), c.ctypes.data_as(_P64),
                         d.ctypes.data_as(_P64), e.ctypes.data_as(_P64),
                         x.ctypes.data_as(_P64), int(LEN_1D))
            return None
    _s2710_nb(a, b, c, d, e, x, LEN_1D)
