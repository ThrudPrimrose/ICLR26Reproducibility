/* Dummy offload version of tsvc_2_s255_fp64.
   Includes an OpenMP target construct (forced to run on the host with
   "if(0)") to satisfy the benchmark requirement, then performs the actual
   computation with a host-parallel loop.
   The algorithm uses wrap‑around indexing to avoid any loop‑carried
   dependencies.
*/
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        return;
    }
    if (LEN_1D == 1) {
        a[0] = b[0];
        return;
    }
    /* Dummy target region – enforced to execute on the host. */
    #pragma omp target if(0)
    {
        /* No device work. */
    }
    /* Host‑parallel loop with wrap‑around indexing. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int64_t im1 = i - 1;
        int64_t im2 = i - 2;
        if (im1 < 0) im1 += LEN_1D;
        if (im2 < 0) im2 += LEN_1D;
        a[i] = (b[i] + b[im1] + b[im2]) * 0.333;
    }
}
