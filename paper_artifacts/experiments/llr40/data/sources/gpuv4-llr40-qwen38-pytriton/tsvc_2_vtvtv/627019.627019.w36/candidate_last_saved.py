"""TSVC tsvc_2 vtvtv: a[i] = a[i] * b[i] * c[i] (in-place, fp64).

Strategy: single-pass Numba kernel (bit-exact op order, no fastmath),
threaded via prange over contiguous per-thread chunks. Everything is
compiled and thread-count-tuned at import time so the timed call is a
warm native call.
"""
import os
import time as _time

import numpy as np

from numba.core import config as _nbcfg
try:
    _nbcfg.CPU_ISA = "AVX512F"
except Exception:
    pass
import numba
from numba import njit, prange


@njit
def _seq(a, b, c, n):
    for i in range(n):
        a[i] = a[i] * b[i] * c[i]


@njit
def _par(a, b, c, n, block):
    nb = (n + block - 1) // block
    for t in prange(nb):
        i0 = t * block
        i1 = i0 + block
        if i1 > n:
            i1 = n
        i = i0
        while i + 8 <= i1:
            a[i]   = a[i]   * b[i]   * c[i]
            a[i+1] = a[i+1] * b[i+1] * c[i+1]
            a[i+2] = a[i+2] * b[i+2] * c[i+2]
            a[i+3] = a[i+3] * b[i+3] * c[i+3]
            a[i+4] = a[i+4] * b[i+4] * c[i+4]
            a[i+5] = a[i+5] * b[i+5] * c[i+5]
            a[i+6] = a[i+6] * b[i+6] * c[i+6]
            a[i+7] = a[i+7] * b[i+7] * c[i+7]
            i += 8
        while i < i1:
            a[i] = a[i] * b[i] * c[i]
            i += 1


def _autotune(verbose=False):
    import sys
    try:
        import numba.core.config as _ncfg
        _cap = int(_ncfg.NUMBA_NUM_THREADS)
    except Exception:
        _cap = 192
    maxt = max(1, min(_cap, int(os.environ.get("VT_TMAX", 192))))
    want = (4, 8, 16, 24, 32, 48, 64, 96, 128, 160)
    cands = [t for t in want if t <= maxt] or [maxt]
    n = 150_000_000
    try:
        d = np.ones(n)
        best_t, best_x = cands[-1], None
        for t in cands:
            numba.set_num_threads(t)
            _par(d, d, d, n, 4096)  # warm / first compile
            t0 = _time.perf_counter()
            _par(d, d, d, n, 4096)
            x = _time.perf_counter() - t0
            if verbose:
                print("sweep T=%d %.4f ms" % (t, x * 1e3), file=sys.stdout, flush=True)
            if best_x is None or x < best_x:
                best_t, best_x = t, x
        numba.set_num_threads(best_t)
        return best_t
    except Exception:
        numba.set_num_threads(min(16, maxt))
        return min(16, maxt)


_T = _autotune(verbose=True)

# Pre-compile the small-N path at import time so no first-call cost.
_sm = np.ones(16)
_seq(_sm, _sm, _sm, 16)


def vtvtv(a, b, c, LEN_1D):
    n = int(LEN_1D)
    if n <= 2_000_000:
        _seq(a, b, c, n)
    else:
        _par(a, b, c, n, 4096)
    return None
