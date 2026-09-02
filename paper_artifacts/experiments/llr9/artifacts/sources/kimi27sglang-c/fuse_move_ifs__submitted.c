#include <stddef.h>
#include <stdint.h>

void fuse_move_ifs_fp64(double * restrict a, double * restrict b, const double * restrict cond, const double * restrict src, int64_t K, int64_t LEN_2D, uint8_t * restrict workspace, int64_t workspace_bytes) {
    if (K > 0) {
        #pragma omp parallel for
        for (int64_t i = 0; i < LEN_2D; i++) {
            if (cond[i] > 0.0) {
                for (int64_t j = 0; j < LEN_2D; j++) {
                    double v = src[i * LEN_2D + j];
                    a[i * LEN_2D + j] = v * 2.0;
                    b[i * LEN_2D + j] = v + 1.0;
                }
            } else {
                for (int64_t j = 0; j < LEN_2D; j++) {
                    b[i * LEN_2D + j] = src[i * LEN_2D + j] + 1.0;
                }
            }
        }
    } else {
        #pragma omp parallel for
        for (int64_t i = 0; i < LEN_2D; i++) {
            if (cond[i] > 0.0) {
                for (int64_t j = 0; j < LEN_2D; j++) {
                    a[i * LEN_2D + j] = src[i * LEN_2D + j] * 2.0;
                }
            }
        }
    }
}
