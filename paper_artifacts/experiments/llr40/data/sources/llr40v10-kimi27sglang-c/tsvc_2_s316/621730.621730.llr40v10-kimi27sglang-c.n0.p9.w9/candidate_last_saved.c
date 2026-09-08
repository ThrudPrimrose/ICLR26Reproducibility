#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline double hmin_avx2(__m256d v) {
    double lanes[4] __attribute__((aligned(32)));
    _mm256_store_pd(lanes, v);
    double m = lanes[0];
    for (int j = 1; j < 4; ++j) {
        if (lanes[j] < m) m = lanes[j];
    }
    return m;
}

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
    double global_min = a[0];

    if (LEN_1D > 1) {
        #pragma omp parallel
        {
            int nt = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int64_t n = LEN_1D - 1;
            int64_t chunk = n / nt;
            int64_t rem = n % nt;
            int64_t start = 1 + tid * chunk + (tid < rem ? tid : rem);
            int64_t end = start + chunk + (tid < rem ? 1 : 0);

            __m256d v0 = _mm256_set1_pd(global_min);
            __m256d v1 = v0;
            __m256d v2 = v0;
            __m256d v3 = v0;
            int64_t i = start;

            for (; i + 16 <= end; i += 16) {
                __m256d a0 = _mm256_loadu_pd(&a[i]);
                __m256d a1 = _mm256_loadu_pd(&a[i + 4]);
                __m256d a2 = _mm256_loadu_pd(&a[i + 8]);
                __m256d a3 = _mm256_loadu_pd(&a[i + 12]);
                __m256d lt0 = _mm256_cmp_pd(a0, v0, _CMP_LT_OQ);
                __m256d lt1 = _mm256_cmp_pd(a1, v1, _CMP_LT_OQ);
                __m256d lt2 = _mm256_cmp_pd(a2, v2, _CMP_LT_OQ);
                __m256d lt3 = _mm256_cmp_pd(a3, v3, _CMP_LT_OQ);
                v0 = _mm256_blendv_pd(v0, a0, lt0);
                v1 = _mm256_blendv_pd(v1, a1, lt1);
                v2 = _mm256_blendv_pd(v2, a2, lt2);
                v3 = _mm256_blendv_pd(v3, a3, lt3);
            }

            for (; i + 4 <= end; i += 4) {
                __m256d va = _mm256_loadu_pd(&a[i]);
                __m256d lt = _mm256_cmp_pd(va, v0, _CMP_LT_OQ);
                v0 = _mm256_blendv_pd(v0, va, lt);
            }

            __m256d lt01 = _mm256_cmp_pd(v1, v0, _CMP_LT_OQ);
            v0 = _mm256_blendv_pd(v0, v1, lt01);
            __m256d lt02 = _mm256_cmp_pd(v2, v0, _CMP_LT_OQ);
            v0 = _mm256_blendv_pd(v0, v2, lt02);
            __m256d lt03 = _mm256_cmp_pd(v3, v0, _CMP_LT_OQ);
            v0 = _mm256_blendv_pd(v0, v3, lt03);

            double local_min = hmin_avx2(v0);

            for (; i < end; ++i) {
                if (a[i] < local_min) {
                    local_min = a[i];
                }
            }

            #pragma omp critical
            {
                if (local_min < global_min) {
                    global_min = local_min;
                }
            }
        }
    }

    result[0] = global_min;
}
