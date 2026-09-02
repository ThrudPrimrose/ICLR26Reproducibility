#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void fuse_move_ifs_fp64(double* a,
                        double* b,
                        double* cond,
                        double* src,
                        int64_t K,
                        int64_t LEN_2D,
                        uint8_t* workspace,
                        int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    const int64_t n = LEN_2D;

    if (K > 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            const int64_t row = i * n;
            if (cond[i] > 0.0) {
                #pragma omp simd
                for (int64_t j = 0; j < n; ++j) {
                    a[row + j] = src[row + j] * 2.0;
                }
            }
            #pragma omp simd
            for (int64_t j = 0; j < n; ++j) {
                b[row + j] = src[row + j] + 1.0;
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            if (cond[i] > 0.0) {
                const int64_t row = i * n;
                #pragma omp simd
                for (int64_t j = 0; j < n; ++j) {
                    a[row + j] = src[row + j] * 2.0;
                }
            }
        }
    }
}
