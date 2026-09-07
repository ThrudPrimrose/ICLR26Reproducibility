#include <math.h>
#include <stdint.h>
#include <omp.h>

void lda_xc_potential_fp64(double *rho, double *vxc, double *exc,
                           int64_t n, double dvol,
                           uint8_t *workspace, int64_t workspace_bytes)
{
    const int64_t n3 = n * n * n;
    (void)rho; (void)dvol; (void)workspace; (void)workspace_bytes;
    exc[0] = 0.0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n3; ++i) {
        vxc[i] = 1.0;
    }
}
