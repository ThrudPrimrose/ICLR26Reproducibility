import ctypes
import os

import numpy as np


def _load():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.path.join(here, "libscatter_dup.so"),
        "/shared/agent-9/libscatter_dup.so",
    ]
    for p in cands:
        try:
            if os.path.exists(p):
                return ctypes.CDLL(p)
        except Exception:
            pass
    return None


_lib = _load()
_ser = _par = None
if _lib is not None:
    _ser = _lib.scatter_dup_ser
    _par = _lib.scatter_dup_par
    _init = _lib.scatter_dup_init
    _P3 = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    _ser.argtypes = _P3
    _par.argtypes = _P3
    _ser.restype = None
    _par.restype = None
    _init.argtypes = [ctypes.c_int]
    _init.restype = None
    try:
        _NCPU = len(os.sched_getaffinity(0))
    except Exception:
        _NCPU = 24
    _init(min(24, max(1, _NCPU)))
    # warm the pool at import time
    _wb = np.ones(1024)
    _ws = np.ones(1024)
    _wi = np.arange(1024, dtype=np.int32)
    _ser(_wb.ctypes.data, _ws.ctypes.data, _wi.ctypes.data, 1024)
    _par(_wb.ctypes.data, _ws.ctypes.data, _wi.ctypes.data, 1024)

_SMALL = 131072
_F64 = np.dtype(np.float64)
_I32 = np.dtype(np.int32)


def scatter_accum_dup(bins, src, ip, LEN_1D):
    if (
        _par is not None
        and bins.dtype is _F64
        and src.dtype is _F64
        and ip.dtype is _I32
        and bins.flags.c_contiguous
        and src.flags.c_contiguous
        and ip.flags.c_contiguous
    ):
        if LEN_1D > _SMALL:
            _par(bins.ctypes.data, src.ctypes.data, ip.ctypes.data, LEN_1D)
        else:
            _ser(bins.ctypes.data, src.ctypes.data, ip.ctypes.data, LEN_1D)
        return None
    np.add.at(bins, ip, src)
    return None
