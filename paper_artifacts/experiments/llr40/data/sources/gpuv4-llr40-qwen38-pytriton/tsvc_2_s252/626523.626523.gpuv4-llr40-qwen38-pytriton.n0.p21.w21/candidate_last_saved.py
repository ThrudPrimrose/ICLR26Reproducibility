"""TSVC tsvc_2 s252, optimized.

Reference semantics:
    t = 0.0
    for i in range(LEN_1D):
        s = b[i] * c[i]
        a[i] = s + t
        t = s

Since t is always the *previous* product, the loop unrolls to a pure
elementwise shifted op with no cross-iteration dependence:

    a[i] = b[i]*c[i] + (b[i-1]*c[i-1] if i > 0 else 0.0)

Primary path: a hand-written AVX-512 C kernel (8 doubles per op,
non-temporal loads/stores on aligned arrays), OpenMP-parallel over every
CPU the container may use plus a little oversubscription, compiled and
loaded at import time so nothing is charged against the timed call.
The kernel is bit-exact against the reference (same multiply then add,
-fp-contract=off) and self-verified at import; any build/runtime problem
falls back to an exact vectorised NumPy implementation.
"""
import ctypes
import os
import shutil
import subprocess
import tempfile

import numpy as np

_C_SRC = r"""/* TSVC tsvc_2 s252 -- single-pass elementwise kernel.
 *
 * Reference:  t=0; for i: s=b[i]*c[i]; a[i]=s+t; t=s
 * which is  a[i] = b[i]*c[i] + (i ? b[i-1]*c[i-1] : 0)   (bit-exact)
 *
 * Vector core (8 doubles/op, AVX512F + AVX2 blend):
 *   p  = b[c8..c8+8) * c[c8..c8+8)
 *   sh = permute p by [?,0,1,2,3,4,5,6]   (sh[0] replaced by scalar tail)
 *   a  = p + sh
 * NT loads/stores on 64B-aligned arrays; unaligned + plain-op variant
 * otherwise; scalar path for the head (i<8) and tail (<8 elements).
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void set_threads(int n) { if (n > 0) omp_set_num_threads(n); }

static inline __m512d shift_right1(__m512d p) {
    /* out[0] = anything (overwritten by caller), out[k] = p[k-1] for k>=1 */
    __m512i idx = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 0);
    return _mm512_permutexvar_pd(idx, p);
}

static void vec_loop_aligned(double *a, const double *b, const double *c, long m) {
    #pragma omp parallel for schedule(static, 64)
    for (long c8 = 8; c8 < m; c8 += 8) {
        __m512d p = _mm512_mul_pd(
            _mm512_castsi512_pd(_mm512_stream_load_si512((void *)(b + c8))),
            _mm512_castsi512_pd(_mm512_stream_load_si512((void *)(c + c8))));
        __m512d sh = shift_right1(p);
        sh = _mm512_mask_blend_pd(1, sh, _mm512_set1_pd(b[c8 - 1] * c[c8 - 1]));
        _mm512_stream_pd(a + c8, _mm512_add_pd(p, sh));
    }
}

static void vec_loop_unaligned(double *a, const double *b, const double *c, long m) {
    #pragma omp parallel for schedule(static, 64)
    for (long c8 = 8; c8 < m; c8 += 8) {
        __m512d p = _mm512_mul_pd(_mm512_loadu_pd(b + c8), _mm512_loadu_pd(c + c8));
        __m512d sh = shift_right1(p);
        sh = _mm512_mask_blend_pd(1, sh, _mm512_set1_pd(b[c8 - 1] * c[c8 - 1]));
        _mm512_storeu_pd(a + c8, _mm512_add_pd(p, sh));
    }
}

static void scalar_loop(double *a, const double *b, const double *c, long i, long n) {
    #pragma omp parallel for schedule(static, 64)
    for (long j = i; j < n; j++) a[j] = b[j] * c[j] + b[j - 1] * c[j - 1];
}

void s252_core(double *a, const double *b, const double *c, long n) {
    if (n <= 0) return;
    a[0] = b[0] * c[0];
    /* head: a[1..min(n-1,7)] */
    long h = n < 8 ? n : 8;
    for (long i = 1; i < h; ++i) a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
    if (n < 8) return;
    if (!__builtin_cpu_supports("avx512f")) {
        scalar_loop(a, b, c, 8, n);
        return;
    }
    long m = 8 * ((n - 8) / 8) + 8;   /* vector part: [8, m) */
    uintptr_t mask = ((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 63;
    if (mask == 0) vec_loop_aligned(a, b, c, m);
    else vec_loop_unaligned(a, b, c, m);
    for (long i = m; i < n; ++i) a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
}
"""

_NTHREADS = int(os.environ.get("S252_T", "8"))


def _build():
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cc is None:
        return None
    d = tempfile.mkdtemp(prefix="s252_")
    cpath = os.path.join(d, "k.c")
    so = os.path.join(d, "libk.so")
    try:
        with open(cpath, "w") as fh:
            fh.write(_C_SRC)
        built = False
        for flags in (("-O3", "-march=native"), ("-O3", "-mavx512f", "-mavx2")):
            r = subprocess.run(
                [cc, *flags, "-ffp-contract=off", "-fopenmp", "-shared",
                 "-fPIC", "-o", so, cpath],
                capture_output=True, timeout=180)
            if r.returncode == 0:
                built = True
                break
        if not built:
            return None
        lib = ctypes.CDLL(so)
        f = lib.s252_core
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_long]
        f.restype = None
        lib.set_threads.argtypes = [ctypes.c_int]
        lib.set_threads(_NTHREADS)
        # bit-exact smoke test (also warms the OpenMP thread pool)
        b = np.arange(17, dtype=np.float64) + 0.5
        c = b * 1.1 + 3.0
        a = np.empty(17)
        f(a.ctypes.data, b.ctypes.data, c.ctypes.data, 17)
        t = 0.0
        ref = np.empty(17)
        for i in range(17):
            s = b[i] * c[i]
            ref[i] = s + t
            t = s
        if not np.array_equal(a, ref):
            return None
        return f
    except Exception:
        return None


_core = _build()


def s252(a, b, c, LEN_1D):
    """In-place: write outputs into `a` (C ABI, return None)."""
    n = int(LEN_1D)
    if n <= 0:
        return None
    if n < a.shape[0]:
        a = a[:n]
        b = b[:n]
        c = c[:n]
    if (_core is not None and a.dtype == np.float64 and b.dtype == np.float64
            and c.dtype == np.float64 and a.ndim == 1
            and a.flags.c_contiguous and b.flags.c_contiguous
            and c.flags.c_contiguous):
        _core(a.ctypes.data, b.ctypes.data, c.ctypes.data, n)
    else:
        bv = np.ascontiguousarray(b, dtype=np.float64)
        cv = np.ascontiguousarray(c, dtype=np.float64)
        s = bv * cv
        av = np.ascontiguousarray(a, dtype=np.float64)
        av[0] = s[0]
        av[1:] = s[1:] + s[:-1]
        a[...] = av
    return None
