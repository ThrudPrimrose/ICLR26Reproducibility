#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <stdbool.h>

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e,
                      const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        const int64_t base = LEN_1D / nthreads;
        const int64_t rem = LEN_1D % nthreads;
        const int64_t start = tid * base + (tid < rem ? tid : rem);
        const int64_t end = start + base + (tid < rem ? 1 : 0);

        int64_t i = start;

        if ((((uintptr_t)&a[start] ^ (uintptr_t)&b[start]) & 31) == 0) {
            while (i < end && (((uintptr_t)&a[i]) & 31) != 0) {
                const double t = d[i] * e[i];
                a[i] += t * c[i];
                b[i] = t;
                ++i;
            }
        }

        const bool aligned = (((uintptr_t)&a[i]) & 31) == 0 && (((uintptr_t)&b[i]) & 31) == 0;
        const int64_t aligned_end = end - ((end - i) & 31);

        if (aligned) {
            for (; i < aligned_end; i += 32) {
                __m256d vd0 = _mm256_loadu_pd(&d[i]);
                __m256d ve0 = _mm256_loadu_pd(&e[i]);
                __m256d vc0 = _mm256_loadu_pd(&c[i]);
                __m256d va0 = _mm256_loadu_pd(&a[i]);
                __m256d vt0 = _mm256_mul_pd(vd0, ve0);
                __m256d vn0 = _mm256_fmadd_pd(vt0, vc0, va0);

                __m256d vd1 = _mm256_loadu_pd(&d[i + 4]);
                __m256d ve1 = _mm256_loadu_pd(&e[i + 4]);
                __m256d vc1 = _mm256_loadu_pd(&c[i + 4]);
                __m256d va1 = _mm256_loadu_pd(&a[i + 4]);
                __m256d vt1 = _mm256_mul_pd(vd1, ve1);
                __m256d vn1 = _mm256_fmadd_pd(vt1, vc1, va1);

                __m256d vd2 = _mm256_loadu_pd(&d[i + 8]);
                __m256d ve2 = _mm256_loadu_pd(&e[i + 8]);
                __m256d vc2 = _mm256_loadu_pd(&c[i + 8]);
                __m256d va2 = _mm256_loadu_pd(&a[i + 8]);
                __m256d vt2 = _mm256_mul_pd(vd2, ve2);
                __m256d vn2 = _mm256_fmadd_pd(vt2, vc2, va2);

                __m256d vd3 = _mm256_loadu_pd(&d[i + 12]);
                __m256d ve3 = _mm256_loadu_pd(&e[i + 12]);
                __m256d vc3 = _mm256_loadu_pd(&c[i + 12]);
                __m256d va3 = _mm256_loadu_pd(&a[i + 12]);
                __m256d vt3 = _mm256_mul_pd(vd3, ve3);
                __m256d vn3 = _mm256_fmadd_pd(vt3, vc3, va3);

                _mm256_stream_pd(&a[i], vn0);
                _mm256_stream_pd(&b[i], vt0);
                _mm256_stream_pd(&a[i + 4], vn1);
                _mm256_stream_pd(&b[i + 4], vt1);
                _mm256_stream_pd(&a[i + 8], vn2);
                _mm256_stream_pd(&b[i + 8], vt2);
                _mm256_stream_pd(&a[i + 12], vn3);
                _mm256_stream_pd(&b[i + 12], vt3);
            }
        } else {
            for (; i < aligned_end; i += 32) {
                __m256d vd0 = _mm256_loadu_pd(&d[i]);
                __m256d ve0 = _mm256_loadu_pd(&e[i]);
                __m256d vc0 = _mm256_loadu_pd(&c[i]);
                __m256d va0 = _mm256_loadu_pd(&a[i]);
                __m256d vt0 = _mm256_mul_pd(vd0, ve0);
                __m256d vn0 = _mm256_fmadd_pd(vt0, vc0, va0);

                __m256d vd1 = _mm256_loadu_pd(&d[i + 4]);
                __m256d ve1 = _mm256_loadu_pd(&e[i + 4]);
                __m256d vc1 = _mm256_loadu_pd(&c[i + 4]);
                __m256d va1 = _mm256_loadu_pd(&a[i + 4]);
                __m256d vt1 = _mm256_mul_pd(vd1, ve1);
                __m256d vn1 = _mm256_fmadd_pd(vt1, vc1, va1);

                __m256d vd2 = _mm256_loadu_pd(&d[i + 8]);
                __m256d ve2 = _mm256_loadu_pd(&e[i + 8]);
                __m256d vc2 = _mm256_loadu_pd(&c[i + 8]);
                __m256d va2 = _mm256_loadu_pd(&a[i + 8]);
                __m256d vt2 = _mm256_mul_pd(vd2, ve2);
                __m256d vn2 = _mm256_fmadd_pd(vt2, vc2, va2);

                __m256d vd3 = _mm256_loadu_pd(&d[i + 12]);
                __m256d ve3 = _mm256_loadu_pd(&e[i + 12]);
                __m256d vc3 = _mm256_loadu_pd(&c[i + 12]);
                __m256d va3 = _mm256_loadu_pd(&a[i + 12]);
                __m256d vt3 = _mm256_mul_pd(vd3, ve3);
                __m256d vn3 = _mm256_fmadd_pd(vt3, vc3, va3);

                _mm256_storeu_pd(&a[i], vn0);
                _mm256_storeu_pd(&b[i], vt0);
                _mm256_storeu_pd(&a[i + 4], vn1);
                _mm256_storeu_pd(&b[i + 4], vt1);
                _mm256_storeu_pd(&a[i + 8], vn2);
                _mm256_storeu_pd(&b[i + 8], vt2);
                _mm256_storeu_pd(&a[i + 12], vn3);
                _mm256_storeu_pd(&b[i + 12], vt3);
            }
        }

        for (; i < end; ++i) {
            const double t = d[i] * e[i];
            a[i] += t * c[i];
            b[i] = t;
        }

        _mm_sfence();
    }
}
