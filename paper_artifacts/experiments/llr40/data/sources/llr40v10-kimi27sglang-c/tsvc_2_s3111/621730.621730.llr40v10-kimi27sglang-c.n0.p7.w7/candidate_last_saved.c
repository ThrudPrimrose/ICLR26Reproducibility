#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline double reduce_add_pd(__m512d v) {
    __m256d lo = _mm512_castpd512_pd256(v);
    __m256d hi = _mm512_extractf64x4_pd(v, 1);
    __m256d s = _mm256_add_pd(lo, hi);
    __m128d s128 = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd(s, 1));
    return _mm_cvtsd_f64(_mm_hadd_pd(s128, s128));
}

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 524288) {
        const __m512d vz = _mm512_setzero_pd();
        __m512d acc0 = vz, acc1 = vz, acc2 = vz, acc3 = vz;
        int64_t i = 0;
        for (; i + 32 <= LEN_1D; i += 32) {
            __m512d v0 = _mm512_loadu_pd(a + i);
            __m512d v1 = _mm512_loadu_pd(a + i + 8);
            __m512d v2 = _mm512_loadu_pd(a + i + 16);
            __m512d v3 = _mm512_loadu_pd(a + i + 24);
            acc0 = _mm512_add_pd(acc0, _mm512_max_pd(v0, vz));
            acc1 = _mm512_add_pd(acc1, _mm512_max_pd(v1, vz));
            acc2 = _mm512_add_pd(acc2, _mm512_max_pd(v2, vz));
            acc3 = _mm512_add_pd(acc3, _mm512_max_pd(v3, vz));
        }
        for (; i + 8 <= LEN_1D; i += 8) {
            __m512d v = _mm512_loadu_pd(a + i);
            acc0 = _mm512_add_pd(acc0, _mm512_max_pd(v, vz));
        }
        if (i < LEN_1D) {
            __mmask8 mask = (__mmask8)((1U << (LEN_1D - i)) - 1U);
            __m512d v = _mm512_maskz_loadu_pd(mask, a + i);
            acc0 = _mm512_add_pd(acc0, _mm512_max_pd(v, vz));
        }
        __m512d s01 = _mm512_add_pd(acc0, acc1);
        __m512d s23 = _mm512_add_pd(acc2, acc3);
        __m512d sall = _mm512_add_pd(s01, s23);
        b[0] = reduce_add_pd(sall);
        return;
    }

    double total = 0.0;

    #pragma omp parallel reduction(+:total)
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        int64_t chunk = LEN_1D / nthreads;
        int64_t rem = LEN_1D % nthreads;
        int64_t start = tid * chunk + (tid < rem ? tid : rem);
        int64_t count = chunk + (tid < rem ? 1 : 0);
        int64_t end = start + count;

        const __m512d vz = _mm512_setzero_pd();
        __m512d acc0 = vz, acc1 = vz, acc2 = vz, acc3 = vz;
        __m512d acc4 = vz, acc5 = vz, acc6 = vz, acc7 = vz;

        int64_t i = start;
        for (; i + 64 <= end; i += 64) {
            __m512d v0 = _mm512_loadu_pd(a + i);
            __m512d v1 = _mm512_loadu_pd(a + i + 8);
            __m512d v2 = _mm512_loadu_pd(a + i + 16);
            __m512d v3 = _mm512_loadu_pd(a + i + 24);
            __m512d v4 = _mm512_loadu_pd(a + i + 32);
            __m512d v5 = _mm512_loadu_pd(a + i + 40);
            __m512d v6 = _mm512_loadu_pd(a + i + 48);
            __m512d v7 = _mm512_loadu_pd(a + i + 56);
            acc0 = _mm512_add_pd(acc0, _mm512_max_pd(v0, vz));
            acc1 = _mm512_add_pd(acc1, _mm512_max_pd(v1, vz));
            acc2 = _mm512_add_pd(acc2, _mm512_max_pd(v2, vz));
            acc3 = _mm512_add_pd(acc3, _mm512_max_pd(v3, vz));
            acc4 = _mm512_add_pd(acc4, _mm512_max_pd(v4, vz));
            acc5 = _mm512_add_pd(acc5, _mm512_max_pd(v5, vz));
            acc6 = _mm512_add_pd(acc6, _mm512_max_pd(v6, vz));
            acc7 = _mm512_add_pd(acc7, _mm512_max_pd(v7, vz));
        }
        for (; i + 8 <= end; i += 8) {
            __m512d v = _mm512_loadu_pd(a + i);
            acc0 = _mm512_add_pd(acc0, _mm512_max_pd(v, vz));
        }
        if (i < end) {
            __mmask8 mask = (__mmask8)((1U << (end - i)) - 1U);
            __m512d v = _mm512_maskz_loadu_pd(mask, a + i);
            acc0 = _mm512_add_pd(acc0, _mm512_max_pd(v, vz));
        }
        __m512d s01 = _mm512_add_pd(acc0, acc1);
        __m512d s23 = _mm512_add_pd(acc2, acc3);
        __m512d s45 = _mm512_add_pd(acc4, acc5);
        __m512d s67 = _mm512_add_pd(acc6, acc7);
        __m512d s0123 = _mm512_add_pd(s01, s23);
        __m512d s4567 = _mm512_add_pd(s45, s67);
        __m512d sall = _mm512_add_pd(s0123, s4567);
        total += reduce_add_pd(sall);
    }

    b[0] = total;
}
