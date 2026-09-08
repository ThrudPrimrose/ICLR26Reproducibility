#include <stdint.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    double *restrict aa = __builtin_assume_aligned(a, 64);
    double *restrict bb = __builtin_assume_aligned(b, 64);
    const double *restrict ccond = __builtin_assume_aligned(cond, 64);
    const double *restrict ssrc = __builtin_assume_aligned(src, 64);

    const int use_parallel = (LEN_2D >= 128);
    if (K > 0) {
        #pragma omp parallel for schedule(static) if(use_parallel)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const int64_t row = i * LEN_2D;
            if (ccond[i] > 0.0) {
                #pragma omp simd
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    const double s = ssrc[row + j];
                    aa[row + j] = s * 2.0;
                    bb[row + j] = s + 1.0;
                }
            } else {
                #pragma omp simd
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    bb[row + j] = ssrc[row + j] + 1.0;
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static) if(use_parallel)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            if (ccond[i] > 0.0) {
                #pragma omp simd
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    aa[i * LEN_2D + j] = ssrc[i * LEN_2D + j] * 2.0;
                }
            }
        }
    }
}
