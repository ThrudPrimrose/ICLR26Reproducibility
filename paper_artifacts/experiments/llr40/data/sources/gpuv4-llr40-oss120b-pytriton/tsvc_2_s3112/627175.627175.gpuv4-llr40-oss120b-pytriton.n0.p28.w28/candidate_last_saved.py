import os
import ctypes

_lib_path = "/shared/agent-28/tsvc_2_s3112.so"
_lib = ctypes.CDLL(_lib_path)
_func = _lib.tsvc_2_s3112_fp64
_func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int64]
_func.restype = None

def s3112(a, b, LEN_1D):
    # Directly call the compiled C function. The benchmark supplies contiguous float64 arrays.
    _func(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
          b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
          ctypes.c_int64(LEN_1D))
