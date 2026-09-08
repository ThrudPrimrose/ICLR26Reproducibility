import os, sys, time, ctypes, tempfile, subprocess
import numpy as np

_C_SRC = r'''
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

static void saxpy_8(double* a, const double* b, long n, double s, int nt_mode) {
    long n8 = n >> 3;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (long k = 0; k < n8; k++) {
            long i = k << 3;
            __m512d v = _mm512_fmadd_pd(_mm512_loadu_pd(b + i),
                                        _mm512_set1_pd(s),
                                        _mm512_loadu_pd(a + i));
            if (nt_mode == 2)
                _mm512_storeu_pd(a + i, v);
            else
                _mm512_stream_pd(a + i, v);
        }
        if (nt_mode == 2) _mm_sfence();
    }
    for (long i = n8 << 3; i < n; i++) a[i] += b[i] * s;
}

static void saxpy_16(double* a, const double* b, long n, double s) {
    long n16 = n >> 4;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (long k = 0; k < n16; k++) {
            long i = k << 4;
            __m512d vs = _mm512_set1_pd(s);
            __m512d va0 = _mm512_loadu_pd(a + i), va1 = _mm512_loadu_pd(a + i + 8);
            __m512d vb0 = _mm512_loadu_pd(b + i), vb1 = _mm512_loadu_pd(b + i + 8);
            _mm512_storeu_pd(a + i,     _mm512_fmadd_pd(vb0, vs, va0));
            _mm512_storeu_pd(a + i + 8, _mm512_fmadd_pd(vb1, vs, va1));
        }
    }
    for (long i = n16 << 4; i < n; i++) a[i] += b[i] * s;
}

void saxpy_c(double* a, const double* b, long n, double s) {
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        saxpy_8(a, b, n, s, 1);
    } else {
        for (long i = 0; i < n; i++) a[i] += b[i] * s;
    }
}
void saxpy_stream(double* a, const double* b, long n, double s) {
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        saxpy_8(a, b, n, s, 2);
    } else {
        for (long i = 0; i < n; i++) a[i] += b[i] * s;
    }
}
void saxpy_2x(double* a, const double* b, long n, double s) {
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        saxpy_16(a, b, n, s);
    } else {
        for (long i = 0; i < n; i++) a[i] += b[i] * s;
    }
}
'''

def _build_lib():
    d = tempfile.mkdtemp(prefix="axp_")
    c = os.path.join(d, "axp.c"); so = os.path.join(d, "libaxp.so")
    with open(c, "w") as f: f.write(_C_SRC)
    subprocess.run(["gcc", "-O3", "-mavx512f", "-mfma", "-fopenmp", "-shared",
                    "-fPIC", "-o", so, c], check=True, capture_output=True)
    lib = ctypes.CDLL(so)
    for nm in ("saxpy_c", "saxpy_stream", "saxpy_2x", "omp_set_num_threads"):
        f = getattr(lib, nm)
        f.restype = None
        if nm != "omp_set_num_threads":
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_double]
        else:
            f.argtypes = [ctypes.c_int]
    return lib, so

_lib, _so = _build_lib()

def vpvts(a, b, LEN_1D, S):
    n = int(LEN_1D)
    pc = time.perf_counter
    out = []
    x = a.copy(); xd = x.ctypes.data
    for nt in (24, 48, 96, 128, 160, 192):
        _lib.omp_set_num_threads(nt)
        t0 = pc(); _lib.saxpy_c(xd, xd, n, 2.0); t1 = pc()
        out.append(f"nt{nt}={24*n/1e9/(t1-t0):.0f}GB/s({1e3*(t1-t0):.0f}ms)")
    t0 = pc(); _lib.saxpy_stream(xd, xd, n, 2.0); t1 = pc()
    out.append(f"st96={24*n/1e9/(t1-t0):.0f}GB/s({1e3*(t1-t0):.0f}ms)")
    t0 = pc(); _lib.saxpy_2x(xd, xd, n, 2.0); t1 = pc()
    out.append(f"16x96={24*n/1e9/(t1-t0):.0f}GB/s({1e3*(t1-t0):.0f}ms)")
    # re-run nt24 best combos
    _lib.omp_set_num_threads(24)
    t0 = pc(); _lib.saxpy_stream(xd, xd, n, 2.0); t1 = pc()
    out.append(f"st24={24*n/1e9/(t1-t0):.0f}GB/s({1e3*(t1-t0):.0f}ms)")
    t0 = pc(); _lib.saxpy_2x(xd, xd, n, 2.0); t1 = pc()
    out.append(f"16x24={24*n/1e9/(t1-t0):.0f}GB/s({1e3*(t1-t0):.0f}ms)")
    print(" ".join(out))
    sys.stdout.flush()
    return None
