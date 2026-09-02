#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

/* TSVC tsvc_2 kernel vtvtv: a[i] = a[i]*b[i]*c[i], elementwise triple product.
 * Memory-bandwidth-bound streaming kernel. Hand-written AVX-512, 4x unroll (12 loads
 * in flight per thread) for high memory-level parallelism, threaded across the grading
 * cores (OpenMP). Each thread streams a contiguous chunk; the head loop aligns the
 * stream to a 64B cache-line boundary so the wide accesses below don't split lines.
 * Unaligned intrinsics are used throughout: they are always correct regardless of the
 * base pointer's alignment and, on Zen, cost the same as aligned when on a boundary.
 * Bit-identical to the reference. */
void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b,
                       const double *restrict c, int64_t LEN_1D) {
    #pragma omp parallel
    {
        const int   nt  = omp_get_num_threads();
        const int   tid = omp_get_thread_num();
        const int64_t base = LEN_1D / nt;
        const int64_t rem  = LEN_1D - base * nt;
        const int64_t lo = (int64_t)tid * base + (tid < rem ? tid : rem);
        const int64_t hi = lo + base + (tid < rem ? 1 : 0);
        int64_t i = lo;

        /* head: scalar until (a+i) is on a 64B cache-line boundary */
        while (i < hi && (uintptr_t)(a + i) % 64u != 0u) {
            a[i] = (a[i] * b[i]) * c[i]; ++i;
        }

        /* main: 4x unrolled AVX-512, 32 doubles/iter */
        for (; i + 32 <= hi; i += 32) {
            __m512d va0 = _mm512_loadu_pd(a + i);
            __m512d va1 = _mm512_loadu_pd(a + i + 8);
            __m512d va2 = _mm512_loadu_pd(a + i + 16);
            __m512d va3 = _mm512_loadu_pd(a + i + 24);
            __m512d vb0 = _mm512_loadu_pd(b + i);
            __m512d vb1 = _mm512_loadu_pd(b + i + 8);
            __m512d vb2 = _mm512_loadu_pd(b + i + 16);
            __m512d vb3 = _mm512_loadu_pd(b + i + 24);
            __m512d vc0 = _mm512_loadu_pd(c + i);
            __m512d vc1 = _mm512_loadu_pd(c + i + 8);
            __m512d vc2 = _mm512_loadu_pd(c + i + 16);
            __m512d vc3 = _mm512_loadu_pd(c + i + 24);
            va0 = _mm512_mul_pd(_mm512_mul_pd(va0, vb0), vc0);
            va1 = _mm512_mul_pd(_mm512_mul_pd(va1, vb1), vc1);
            va2 = _mm512_mul_pd(_mm512_mul_pd(va2, vb2), vc2);
            va3 = _mm512_mul_pd(_mm512_mul_pd(va3, vb3), vc3);
            _mm512_storeu_pd(a + i, va0);
            _mm512_storeu_pd(a + i + 8, va1);
            _mm512_storeu_pd(a + i + 16, va2);
            _mm512_storeu_pd(a + i + 24, va3);
        }
        /* tail: 8-wide then scalar */
        for (; i + 8 <= hi; i += 8) {
            __m512d va = _mm512_loadu_pd(a + i);
            va = _mm512_mul_pd(_mm512_mul_pd(va, _mm512_loadu_pd(b + i)),
                               _mm512_loadu_pd(c + i));
            _mm512_storeu_pd(a + i, va);
        }
        for (; i < hi; ++i) { a[i] = (a[i] * b[i]) * c[i]; }
    }
}
