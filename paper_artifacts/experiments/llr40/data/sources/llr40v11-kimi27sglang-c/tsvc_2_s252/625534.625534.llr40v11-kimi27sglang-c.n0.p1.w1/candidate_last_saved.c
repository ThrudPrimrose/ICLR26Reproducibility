#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

__attribute__((noinline, optimize("no-tree-vectorize")))
static void scalar_serial(double *restrict a,
                          const double *restrict b,
                          const double *restrict c,
                          int64_t n)
{
    double prev = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double cur = b[i] * c[i];
        a[i] = cur + prev;
        prev = cur;
    }
}

static inline void simd_serial(double *restrict a,
                               const double *restrict b,
                               const double *restrict c,
                               int64_t n)
{
    double prev = 0.0;
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_loadu_pd(c + i);
        __m256d cur = _mm256_mul_pd(vb, vc);

        __m256d shifted = _mm256_permute4x64_pd(cur, _MM_SHUFFLE(2, 1, 0, 3));
        __m256d prev_bc = _mm256_set1_pd(prev);
        __m256d addend = _mm256_blend_pd(shifted, prev_bc, 1);
        __m256d out = _mm256_add_pd(cur, addend);
        _mm256_storeu_pd(a + i, out);

        prev_bc = _mm256_permute4x64_pd(cur, _MM_SHUFFLE(3, 3, 3, 3));
        prev = _mm256_cvtsd_f64(prev_bc);
    }
    for (; i < n; ++i) {
        double cur = b[i] * c[i];
        a[i] = cur + prev;
        prev = cur;
    }
}

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    if (LEN_1D < 4096) {
        scalar_serial(a, b, c, LEN_1D);
        return;
    }

    int max_threads = omp_get_max_threads();
    if (max_threads <= 1) {
        simd_serial(a, b, c, LEN_1D);
        return;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int64_t chunk = (LEN_1D + nt - 1) / nt;
        int64_t start = tid * chunk;
        int64_t end = start + chunk;
        if (start > LEN_1D) start = LEN_1D;
        if (end > LEN_1D) end = LEN_1D;

        if (start < end) {
            double prev = (start > 0) ? b[start - 1] * c[start - 1] : 0.0;
            int64_t i = start;

            __m256d prev_bc = _mm256_set1_pd(prev);
            for (; i + 3 < end; i += 4) {
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vc = _mm256_loadu_pd(c + i);
                __m256d cur = _mm256_mul_pd(vb, vc);

                __m256d shifted = _mm256_permute4x64_pd(cur, _MM_SHUFFLE(2, 1, 0, 3));
                __m256d addend = _mm256_blend_pd(shifted, prev_bc, 1);
                __m256d out = _mm256_add_pd(cur, addend);
                _mm256_storeu_pd(a + i, out);

                prev_bc = _mm256_permute4x64_pd(cur, _MM_SHUFFLE(3, 3, 3, 3));
            }
            prev = _mm256_cvtsd_f64(prev_bc);
            for (; i < end; ++i) {
                double cur = b[i] * c[i];
                a[i] = cur + prev;
                prev = cur;
            }
        }
    }
}
