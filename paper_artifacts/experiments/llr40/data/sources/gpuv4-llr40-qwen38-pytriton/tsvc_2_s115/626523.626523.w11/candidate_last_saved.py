"""Optimized tsvc_2_s115: a[i] -= sum_{j<i} aa[j, i] * a[j]  (in-place on a).

Strategy:
  * small N: single C kernel (built at import; numba fallback)
  * large N: blocked -- past part of each block via multithreaded BLAS dgemv,
    in-block triangular part via a small numba kernel.
"""
import sys
import os
import subprocess
import tempfile
import importlib.machinery
import importlib.util

import numpy as np
import numba as nb

_B = 256          # block size for the dgemv path
_SMALL_THRESH = 1500

# ---------------- numba kernels ----------------

@nb.njit(boundscheck=False)
def _numba_small(a, af, N):
    for j in range(N):
        aj = a[j]
        base = j * N
        for i in range(j + 1, N):
            a[i] -= af[base + i] * aj


@nb.njit(boundscheck=False)
def _numba_inblock(a, af, i0, i1, N):
    for i in range(i0, i1):
        ai = a[i]
        for j in range(i0, i):
            ai -= af[j * N + i] * a[j]
        a[i] = ai


# ---------------- optional C fast path for small N ----------------

_C_SRC = r"""
#include <stdint.h>
#include <stddef.h>

void tsvc_s115_small_c(double *a, const double *aa, int64_t N) {
    for (int64_t j = 0; j < N; j++) {
        const double aj = a[j];
        const double *aaj = aa + j * (size_t)N;
        for (int64_t i = j + 1; i < N; i++) {
            a[i] -= aaj[i] * aj;
        }
    }
}
"""

_SMALL_FN = None  # callable(a, aa, N) -> None


def _build_small_c():
    global _SMALL_FN
    d = tempfile.mkdtemp(prefix="tsvc115_")
    cpath = os.path.join(d, "k.c")
    solib = os.path.join(d, "tsvc115_smallk.so")
    with open(cpath, "w") as f:
        f.write(_C_SRC)
    last_err = None
    cflags = (["-O3", "-march=native"], ["-O2"], ["-O1"])
    compilers = (["gcc"], ["cc"], ["/usr/bin/gcc"])
    for comp in compilers:
        for flags in cflags:
            try:
                r = subprocess.run(
                    comp + ["-shared", "-fPIC"] + flags + [cpath, "-o", solib],
                    capture_output=True, timeout=120,
                )
                if r.returncode == 0:
                    import ctypes
                    lib = ctypes.CDLL(solib)
                    fn = lib.tsvc_s115_small_c
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
                    fn.restype = None

                    def _call(a, aa, N, _fn=fn):
                        _fn(a.ctypes.data, aa.ctypes.data, N)

                    # sanity check before adopting
                    n = 37
                    a0 = np.arange(n, dtype=np.float64)
                    a = a0.copy()
                    aa = np.ones((n, n))
                    _call(a, aa, n)
                    ref = a0.copy()
                    for j in range(n):
                        for i in range(j + 1, n):
                            ref[i] -= aa[j, i] * ref[j]
                    if not np.array_equal(a, ref):
                        return None
                    return _call
            except Exception as e:  # noqa
                last_err = e
    return None


def _warmup():
    global _SMALL_FN
    try:
        _SMALL_FN = _build_small_c()
    except Exception:
        _SMALL_FN = None

    rng = np.random.default_rng(0)
    if _SMALL_FN is None:
        _SMALL_FN = _numba_small
        a = rng.standard_normal(64)
        aa = rng.standard_normal((64, 64))
        _numba_small(a, aa.reshape(-1), 64)

    # warm numba in-block + dgemv path (BLAS thread pool)
    n = 2048
    a = rng.standard_normal(n)
    aa = rng.standard_normal((n, n))
    s115(a, aa, n)


def s115(a, aa, LEN_2D):
    N = int(LEN_2D)
    if N <= 1:
        return None
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    if not aa.flags["C_CONTIGUOUS"]:
        aa = np.ascontiguousarray(aa)
    if N < _SMALL_THRESH:
        _SMALL_FN(a, aa.reshape(-1), N)
        return None
    af = aa.reshape(-1)
    B = _B
    for i0 in range(0, N, B):
        i1 = min(i0 + B, N)
        # dgemv first: finalizes a[i0] (its only pending contribution)
        if i0:
            a[i0:i1] -= np.dot(aa[:i0, i0:i1].T, a[:i0])
        if i1 > i0 + 1:
            _numba_inblock(a, af, i0, i1, N)
    return None


_warmup()
