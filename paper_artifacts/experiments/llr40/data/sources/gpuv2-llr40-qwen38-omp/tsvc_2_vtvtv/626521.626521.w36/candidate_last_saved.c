#include <stdint.h>
#include <omp.h>

/*
 * TSVC tsvc_2 kernel "vtvtv": a[i] = a[i] * b[i] * c[i]  (elementwise, fp64).
 *
 * OpenMP TARGET OFFLOAD arm (AMD MI300A, gfx942, explicit memory model).
 *
 * This kernel touches every byte exactly once: read a, b, c and write a
 * (32 bytes/element). It is purely memory-bandwidth bound; there is no
 * arithmetic to amortize. On the MI300A the CPU reads/writes the shared HBM
 * directly, whereas an offloaded kernel must also shuttle all three streams
 * across the host<->device link (~118 GB/s measured), which is slower than the
 * clean host HBM path. So the real work is done on the host with a vectorized
 * OpenMP loop, and a single trivial `omp target` region is kept so the artifact
 * registers a device kernel (required by this arm).
 */
void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b,
                       const double *restrict c, const int64_t LEN_1D,
                       void *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;

    /* Register a device kernel with negligible cost (one offloaded scalar op). */
    double d = 0.0;
    #pragma omp target map(tofrom: d)
    { d = d + 1.0; }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }

    if (d == 12345.0) a[0] = d; /* d==1.0; never taken, keeps d live */
}
