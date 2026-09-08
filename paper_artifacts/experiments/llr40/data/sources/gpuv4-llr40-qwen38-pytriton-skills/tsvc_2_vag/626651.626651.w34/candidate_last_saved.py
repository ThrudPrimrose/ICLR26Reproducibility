import numpy as np
import numba as nb
from numba import prange


@nb.njit(parallel=True, fastmath=True)
def _u8(a, b, ip, n):
    m8 = n // 8
    for c in prange(m8):
        i = c * 8
        a[i] = b[ip[i]]
        a[i + 1] = b[ip[i + 1]]
        a[i + 2] = b[ip[i + 2]]
        a[i + 3] = b[ip[i + 3]]
        a[i + 4] = b[ip[i + 4]]
        a[i + 5] = b[ip[i + 5]]
        a[i + 6] = b[ip[i + 6]]
        a[i + 7] = b[ip[i + 7]]
    for i in range(m8 * 8, n):
        a[i] = b[ip[i]]


def vag(a, b, ip, LEN_1D):
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    _u8(a, b, ip, int(LEN_1D))
    return None


tsvc_2_vag = vag


def _warm():
    a = np.zeros(1024, dtype=np.float64)
    b = np.ones(1024, dtype=np.float64)
    ip = np.arange(1024, dtype=np.int32)
    _u8(a, b, ip, 1024)
    _u8(a, b, ip, 1023)


_warm()
