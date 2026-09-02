#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline double fma_d(double x, double y, double z) {
    return __builtin_fma(x, y, z);
}

static void run_single_thread(double *restrict a, const double *restrict b, const double *restrict c,
                              int64_t LEN_1D) {
    int64_t i = 0;

    for (; i + 16 <= LEN_1D; i += 16) {
        __m512d av0 = _mm512_loadu_pd(&a[i]);
        __m512d bv0 = _mm512_loadu_pd(&b[i]);
        __m512d cv0 = _mm512_loadu_pd(&c[i]);
        __m512d res0 = _mm512_fmadd_pd(bv0, cv0, av0);
        __mmask8 mask0 = _mm512_cmp_pd_mask(cv0, bv0, _CMP_GT_OQ);

        __m512d av1 = _mm512_loadu_pd(&a[i + 8]);
        __m512d bv1 = _mm512_loadu_pd(&b[i + 8]);
        __m512d cv1 = _mm512_loadu_pd(&c[i + 8]);
        __m512d res1 = _mm512_fmadd_pd(bv1, cv1, av1);
        __mmask8 mask1 = _mm512_cmp_pd_mask(cv1, bv1, _CMP_GT_OQ);

        if (__builtin_expect((mask0 | mask1) != 0, 0)) {
            if (__builtin_expect(mask0 != 0, 0)) {
                int first = __builtin_ctz((unsigned int)mask0);
                int64_t end = i + first;
                for (int64_t j = i; j <= end; ++j) {
                    a[j] = fma_d(b[j], c[j], a[j]);
                }
            } else {
                _mm512_storeu_pd(&a[i], res0);
                int first = __builtin_ctz((unsigned int)mask1);
                int64_t end = i + 8 + first;
                for (int64_t j = i + 8; j <= end; ++j) {
                    a[j] = fma_d(b[j], c[j], a[j]);
                }
            }
            return;
        }
        _mm512_storeu_pd(&a[i], res0);
        _mm512_storeu_pd(&a[i + 8], res1);
    }

    for (; i + 8 <= LEN_1D; i += 8) {
        __m512d av = _mm512_loadu_pd(&a[i]);
        __m512d bv = _mm512_loadu_pd(&b[i]);
        __m512d cv = _mm512_loadu_pd(&c[i]);
        __m512d res = _mm512_fmadd_pd(bv, cv, av);
        __mmask8 mask = _mm512_cmp_pd_mask(cv, bv, _CMP_GT_OQ);
        if (__builtin_expect(mask != 0, 0)) {
            int first = __builtin_ctz((unsigned int)mask);
            int64_t end = i + first;
            for (int64_t j = i; j <= end; ++j) {
                a[j] = fma_d(b[j], c[j], a[j]);
            }
            return;
        }
        _mm512_storeu_pd(&a[i], res);
    }

    for (; i + 4 <= LEN_1D; i += 4) {
        __m256d av = _mm256_loadu_pd(&a[i]);
        __m256d bv = _mm256_loadu_pd(&b[i]);
        __m256d cv = _mm256_loadu_pd(&c[i]);
        __m256d res = _mm256_fmadd_pd(bv, cv, av);
        __m256d maskv = _mm256_cmp_pd(cv, bv, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(maskv);
        if (__builtin_expect(mask != 0, 0)) {
            int first = __builtin_ctz(mask);
            int64_t end = i + first;
            for (int64_t j = i; j <= end; ++j) {
                a[j] = fma_d(b[j], c[j], a[j]);
            }
            return;
        }
        _mm256_storeu_pd(&a[i], res);
    }

    for (; i < LEN_1D; ++i) {
        a[i] = fma_d(b[i], c[i], a[i]);
        if (c[i] > b[i]) {
            break;
        }
    }
}

void ext_break_post_body_fp64(double *restrict a, const double *restrict b, const double *restrict c,
                              int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_size) {
    if (LEN_1D < 500000) {
        run_single_thread(a, b, c, LEN_1D);
        return;
    }

    int64_t half = LEN_1D / 2;
    int64_t cut = LEN_1D;

    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();

        // Update first half [0, half) unconditionally.
        int64_t u0 = ((int64_t)tid * half) / nt;
        int64_t u1 = ((int64_t)(tid + 1) * half) / nt;
        for (int64_t i = u0; i < u1; ++i) {
            a[i] = fma_d(b[i], c[i], a[i]);
        }

        // Search for the first break in the second half [half, LEN_1D).
        int64_t s0 = half + ((int64_t)tid * (LEN_1D - half)) / nt;
        int64_t s1 = half + ((int64_t)(tid + 1) * (LEN_1D - half)) / nt;
        int64_t local_cut = LEN_1D;
        for (int64_t i = s0; i < s1; ++i) {
            if (c[i] > b[i] && i < local_cut) {
                local_cut = i;
            }
        }
        #pragma omp critical
        {
            if (local_cut < cut) {
                cut = local_cut;
            }
        }

        #pragma omp barrier

        // Update second half only up to and including the break point.
        int64_t v1 = (cut < s1) ? cut + 1 : s1;
        for (int64_t i = s0; i < v1; ++i) {
            a[i] = fma_d(b[i], c[i], a[i]);
        }
    }
}
