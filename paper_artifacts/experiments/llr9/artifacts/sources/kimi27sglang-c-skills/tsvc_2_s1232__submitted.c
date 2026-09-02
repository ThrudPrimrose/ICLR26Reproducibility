#include <stdint.h>
#include <omp.h>

void tsvc_2_s1232_fp64(double *restrict aa, double *restrict bb,
                       double *restrict cc, int64_t LEN_2D, int64_t VLEN,
                       uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    #pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < LEN_2D; i++) {
        const int64_t jmax = i / VLEN;
        for (int64_t j = 0; j <= jmax; j++) {
            aa[i * LEN_2D + j] = bb[i * LEN_2D + j] + cc[i * LEN_2D + j];
        }
    }
}
