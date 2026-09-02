// TSVC quasi_affine_reduce_odd (fp64) -- optimized.
// out[0] = sum(a[i] for i in range(1, LEN_1D, 2))  (odd indices).
//
// Odd elements are interleaved with even ones, so reading only the odd bytes
// still pulls full 64-byte cache lines from DRAM.  We therefore stream the
// array at full width: one masked 512-bit load per 64-byte line keeps only
// the four odd lanes (0xAA = lanes 1,3,5,7) and adds them to a SIMD
// accumulator.  The inner loop is unrolled 8x with independent accumulators
// so several 64-byte loads are in flight at once.  Threads own disjoint
// contiguous block ranges; each stores a partial (into a zeroed array) that
// is combined in a fixed order, so the result is fully deterministic and the
// sum is only over the partials that were produced.  This saturates memory
// bandwidth vs the serial 16-byte-stride reference.
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out,
                                  int64_t LEN_1D) {
    const int64_t n8 = LEN_1D / 8;      /* full 8-double (64-byte) blocks     */
    const int64_t tail_start = 8 * n8;  /* first index not covered by a block */
    int nt = omp_get_max_threads();
    if (nt < 1) nt = 1;
    if (nt > 1024) nt = 1024;
    double partials[1024];
    memset(partials, 0, sizeof(partials));

#if defined(__AVX512F__)
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int ntt = omp_get_num_threads();
        const int64_t k0 = (n8 * (int64_t)tid) / (int64_t)ntt;
        const int64_t k1 = (n8 * (int64_t)(tid + 1)) / (int64_t)ntt;
        const __mmask8 ODD = 0xAA;      /* lanes 1,3,5,7 of each 64-byte line */
        __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
        __m512d a2 = _mm512_setzero_pd(), a3 = _mm512_setzero_pd();
        __m512d a4 = _mm512_setzero_pd(), a5 = _mm512_setzero_pd();
        __m512d a6 = _mm512_setzero_pd(), a7 = _mm512_setzero_pd();
        int64_t k = k0;
        int64_t ke = k1 - ((k1 - k0) & 7);
        for (; k < ke; k += 8) {
            const double *p = a + 8 * k;
            a0 = _mm512_add_pd(a0, _mm512_maskz_loadu_pd(ODD, p));
            a1 = _mm512_add_pd(a1, _mm512_maskz_loadu_pd(ODD, p + 8));
            a2 = _mm512_add_pd(a2, _mm512_maskz_loadu_pd(ODD, p + 16));
            a3 = _mm512_add_pd(a3, _mm512_maskz_loadu_pd(ODD, p + 24));
            a4 = _mm512_add_pd(a4, _mm512_maskz_loadu_pd(ODD, p + 32));
            a5 = _mm512_add_pd(a5, _mm512_maskz_loadu_pd(ODD, p + 40));
            a6 = _mm512_add_pd(a6, _mm512_maskz_loadu_pd(ODD, p + 48));
            a7 = _mm512_add_pd(a7, _mm512_maskz_loadu_pd(ODD, p + 56));
        }
        for (; k < k1; ++k)
            a0 = _mm512_add_pd(a0, _mm512_maskz_loadu_pd(ODD, a + 8 * k));
        __m512d acc = (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)));
        double partial = _mm512_reduce_add_pd(acc);
        if (tid == ntt - 1)
            for (int64_t i = tail_start + 1; i < LEN_1D; i += 2) partial += a[i];
        partials[tid] = partial;
    }
#else
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int ntt = omp_get_num_threads();
        const int64_t k0 = (n8 * (int64_t)tid) / (int64_t)ntt;
        const int64_t k1 = (n8 * (int64_t)(tid + 1)) / (int64_t)ntt;
        double partial = 0.0;
        for (int64_t k = k0; k < k1; ++k) {
            const double *p = a + 8 * k;
            partial += p[1] + p[3] + p[5] + p[7];
        }
        if (tid == ntt - 1)
            for (int64_t i = tail_start + 1; i < LEN_1D; i += 2) partial += a[i];
        partials[tid] = partial;
    }
#endif

    double total = 0.0;
    for (int t = 0; t < nt; ++t) total += partials[t];
    out[0] = total;
}
