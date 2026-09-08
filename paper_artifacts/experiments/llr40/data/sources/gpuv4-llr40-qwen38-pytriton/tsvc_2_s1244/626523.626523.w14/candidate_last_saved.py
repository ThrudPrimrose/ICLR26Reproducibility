import os

_NT = int(os.environ.get("T14", "32"))

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _core(a, b, c, d, n1):
    # d[i] must equal new_a[i] + original_a[i+1]; the original a[i+1] is
    # needed before any thread may overwrite a[i+1], so save it first.
    for i in prange(n1):
        d[i] = a[i + 1]
    for i in prange(n1):
        x = b[i] + c[i] * c[i]
        x = x + b[i] * b[i]
        x = x + c[i]
        a[i] = x
        d[i] = d[i] + x


def _fallback(a, b, c, d, n1):
    cc = c[:n1] * c[:n1]
    bb = b[:n1] * b[:n1]
    new_a = (b[:n1] + cc) + bb
    new_a += c[:n1]
    d[:n1] = new_a + a[1:]
    a[:n1] = new_a


def s1244(a, b, c, d, LEN_1D):
    n1 = LEN_1D - 1
    if n1 <= 0:
        return None
    if a.dtype is np.float64:
        _core(a, b, c, d, n1)
    else:
        _fallback(a, b, c, d, n1)
    return None


# Import-time pre-warm: JIT compile and thread-pool startup happen before
# the timed call.
def _warm():
    try:
        from numba.np.ufunc import parallel as _par
        _par.set_num_threads(_NT)
    except Exception:
        pass
    m = 1 << 20
    a = np.zeros(m)
    b = np.random.default_rng(0).standard_normal(m)
    c = np.random.default_rng(1).standard_normal(m)
    d = np.zeros(m)
    _core(a, b, c, d, m - 1)
    _core(a, b, c, d, 1)


_warm()
