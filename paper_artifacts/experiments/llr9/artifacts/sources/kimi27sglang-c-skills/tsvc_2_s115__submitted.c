#include <stdint.h>
#include <omp.h>

void tsvc_2_s115_fp64(double *restrict a, double *restrict aa, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    const int64_t BS = 128;
    if (LEN_2D <= 256) {
        #pragma omp parallel
        {
            for (int64_t j = 0; j < LEN_2D; ++j) {
                const double aj = a[j];
                #pragma omp for simd schedule(static)
                for (int64_t i = j + 1; i < LEN_2D; ++i) {
                    a[i] -= aa[j * LEN_2D + i] * aj;
                }
            }
        }
        return;
    }
    #pragma omp parallel
    {
        for (int64_t jb = 0; jb < LEN_2D; jb += BS) {
            int64_t je = jb + BS < LEN_2D ? jb + BS : LEN_2D;
            #pragma omp single
            {
                for (int64_t j = jb; j < je; ++j) {
                    const double aj = a[j];
                    for (int64_t i = j + 1; i < je; ++i) {
                        a[i] -= aa[j * LEN_2D + i] * aj;
                    }
                }
            }
            #pragma omp for simd schedule(guided)
            for (int64_t i = je; i < LEN_2D; ++i) {
                for (int64_t j = jb; j < je; ++j) {
                    a[i] -= aa[j * LEN_2D + i] * a[j];
                }
            }
        }
    }
}
