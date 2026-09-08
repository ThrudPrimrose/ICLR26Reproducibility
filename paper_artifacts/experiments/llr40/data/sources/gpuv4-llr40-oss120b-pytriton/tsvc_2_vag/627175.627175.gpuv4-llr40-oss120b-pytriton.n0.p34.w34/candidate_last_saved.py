import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _vag_impl(a, b, ip):
    for i in prange(a.shape[0]):
        a[i] = b[ip[i]]

def vag(a, b, ip, LEN_1D):
    _vag_impl(a, b, ip)
    return None
