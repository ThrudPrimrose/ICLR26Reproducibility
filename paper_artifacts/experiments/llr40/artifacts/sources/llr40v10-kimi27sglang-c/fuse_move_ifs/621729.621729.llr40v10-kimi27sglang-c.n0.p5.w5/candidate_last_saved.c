#include <stdint.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a_, double *restrict b_, const double *restrict cond_, const double *restrict src_,
                        const int64_t K, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    double *restrict a = __builtin_assume_aligned(a_, 64);
    double *restrict b = __builtin_assume_aligned(b_, 64);
    const double *restrict cond = __builtin_assume_aligned(cond_, 64);
    const double *restrict src = __builtin_assume_aligned(src_, 64);

    if (K > 0) {
        #pragma omp parallel for schedule(guided) if (n > 64)
        for (int64_t i = 0; i < n; ++i) {
            const double c = cond[i];
            if (__builtin_expect(c > 0.0, 1)) {
                #pragma omp simd
                for (int64_t j = 0; j < n; ++j) {
                    const double s = src[i * n + j];
                    a[i * n + j] = s * 2.0;
                    b[i * n + j] = s + 1.0;
                }
            } else {
                #pragma omp simd
                for (int64_t j = 0; j < n; ++j) {
                    b[i * n + j] = src[i * n + j] + 1.0;
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(guided) if (n > 64)
        for (int64_t i = 0; i < n; ++i) {
            if (__builtin_expect(cond[i] > 0.0, 1)) {
                #pragma omp simd
                for (int64_t j = 0; j < n; ++j) {
                    a[i * n + j] = src[i * n + j] * 2.0;
                }
            }
        }
    }
}
