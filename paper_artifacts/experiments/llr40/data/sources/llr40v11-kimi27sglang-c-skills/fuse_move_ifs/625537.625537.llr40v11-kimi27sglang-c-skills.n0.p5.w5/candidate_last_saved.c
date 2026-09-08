#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b,
                        const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vtwo = _mm256_set1_pd(2.0);

    if (K > 0) {
        #pragma omp parallel for
        for (int64_t i = 0; i < LEN_2D; ++i) {
            double *restrict row_a = a + i * LEN_2D;
            double *restrict row_b = b + i * LEN_2D;
            const double *restrict row_src = src + i * LEN_2D;
            bool ca = cond[i] > 0.0;

            int64_t j = 0;
            while ((j < LEN_2D) && (((uintptr_t)(row_b + j) & 0x1f) != 0)) {
                double s = row_src[j];
                row_b[j] = s + 1.0;
                if (ca) row_a[j] = s * 2.0;
                ++j;
            }

            if (ca) {
                for (; j + 4 <= LEN_2D; j += 4) {
                    __m256d s = _mm256_loadu_pd(row_src + j);
                    __m256d bv = _mm256_add_pd(s, vone);
                    _mm256_stream_pd(row_b + j, bv);
                    __m256d av = _mm256_mul_pd(s, vtwo);
                    _mm256_storeu_pd(row_a + j, av);
                }
            } else {
                for (; j + 4 <= LEN_2D; j += 4) {
                    __m256d s = _mm256_loadu_pd(row_src + j);
                    __m256d bv = _mm256_add_pd(s, vone);
                    _mm256_stream_pd(row_b + j, bv);
                }
            }

            for (; j < LEN_2D; ++j) {
                double s = row_src[j];
                row_b[j] = s + 1.0;
                if (ca) row_a[j] = s * 2.0;
            }
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < LEN_2D; ++i) {
            if (cond[i] > 0.0) {
                double *restrict row_a = a + i * LEN_2D;
                const double *restrict row_src = src + i * LEN_2D;
                int64_t j = 0;
                while ((j < LEN_2D) && (((uintptr_t)(row_a + j) & 0x1f) != 0)) {
                    row_a[j] = row_src[j] * 2.0;
                    ++j;
                }
                for (; j + 4 <= LEN_2D; j += 4) {
                    __m256d s = _mm256_loadu_pd(row_src + j);
                    __m256d av = _mm256_mul_pd(s, vtwo);
                    _mm256_storeu_pd(row_a + j, av);
                }
                for (; j < LEN_2D; ++j) {
                    row_a[j] = row_src[j] * 2.0;
                }
            }
        }
    }

    _mm_sfence();
}
