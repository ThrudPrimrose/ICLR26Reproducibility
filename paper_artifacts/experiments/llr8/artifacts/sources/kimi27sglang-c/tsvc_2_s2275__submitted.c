#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double* a, double* aa, double* b, double* bb, double* c, double* cc, double* d,
                       int64_t LEN_2D, uint8_t* workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < LEN_2D; ++j) {
        for (int64_t i = 0; i < LEN_2D; ++i) {
            aa[j * LEN_2D + i] += bb[j * LEN_2D + i] * cc[j * LEN_2D + i];
        }
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }
}
