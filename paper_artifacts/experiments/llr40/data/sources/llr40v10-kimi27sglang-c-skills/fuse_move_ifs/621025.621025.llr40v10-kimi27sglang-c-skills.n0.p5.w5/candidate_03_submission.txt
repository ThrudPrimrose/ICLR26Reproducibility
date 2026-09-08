#include <stdint.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    if (K > 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const double c = cond[i];
            double *restrict row_a = &a[i * LEN_2D];
            double *restrict row_b = &b[i * LEN_2D];
            const double *restrict row_src = &src[i * LEN_2D];
            if (c > 0.0) {
                #pragma omp simd
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    const double s = row_src[j];
                    row_a[j] = s * 2.0;
                    row_b[j] = s + 1.0;
                }
            } else {
                #pragma omp simd
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    row_b[j] = row_src[j] + 1.0;
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            if (cond[i] > 0.0) {
                double *restrict row_a = &a[i * LEN_2D];
                const double *restrict row_src = &src[i * LEN_2D];
                #pragma omp simd
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    row_a[j] = row_src[j] * 2.0;
                }
            }
        }
    }
}
