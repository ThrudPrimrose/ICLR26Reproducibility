"""s152: b = d*e; a += b*c. Numba fused single pass, prange for large n. Bit-exact."""
import os
import numpy as np
import numba as nb

def _cap_threads():
    try:
        import numba.np.ufunc.parallel as _npuf
        default = _npuf.get_num_threads()
        _npuf.set_num_threads(min(32, default))
    except Exception:
        pass


_cap_threads()

_PAR_MIN = 1 << 20  # parallelize at >= 1M elements


@nb.njit()
def _s152_ser(a, b, c, d, e, n):
    for i in range(n):
        b[i] = d[i] * e[i]
        a[i] += b[i] * c[i]


@nb.njit(parallel=True)
def _s152_par(a, b, c, d, e, n):
    for i in nb.prange(n):
        b[i] = d[i] * e[i]
        a[i] += b[i] * c[i]


def s152(a, b, c, d, e, LEN_1D):
    n = int(LEN_1D)
    if n >= _PAR_MIN:
        _s152_par(a, b, c, d, e, n)
    else:
        _s152_ser(a, b, c, d, e, n)


def _warm():
    n = 1 << 10
    a = np.ones(n); b = np.ones(n); c = np.ones(n); d = np.ones(n); e = np.ones(n)
    _s152_ser(a, b, c, d, e, n)
    _s152_par(a, b, c, d, e, n)


_warm()
