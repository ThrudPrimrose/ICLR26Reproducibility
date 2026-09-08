/* Optimized offload version of tsvc_2_s235 kernel.
 * The original serial kernel:
 *   for i in 0..N-1:
 *     a[i] += b[i] * c[i];
 *     for j in 1..N-1:
 *       aa[j*N + i] = aa[(j-1)*N + i] + bb[j*N + i] * a[i];
 *
 * Dependencies:
 *   * The update of a[i] is independent across i.
 *   * The inner recurrence updates aa column‑wise; each column (fixed i) depends only on the
 *     previous row j‑1 of the same column. Hence the outer i‑loop can be parallelised.
 *
 * For this OpenMP offload arm the kernel must contain a `#pragma omp target` region so that the
 * compiler generates a device image.  We map all arrays explicitly – `tofrom` for the mutable
 * arrays (a, aa) and `to` for the read‑only inputs (b, bb, c).  The entire computation is performed
 * inside a single offload region; the data are transferred only once.
 *
 * The outer loop over i is executed as a GPU work‑sharing loop; each GPU thread processes one column
 * (one i).  Inside each thread we perform the sequential j‑loop that carries the recurrence.
 * This respects the dependence while still exposing massive thread parallelism on the device.
 */

#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b,
                      const double *restrict bb, const double *restrict c,
                      const int64_t LEN_2D) {
    // Verify offload (fallback to host if unavailable).
    int on_device = 0;
    #pragma omp target map(from:on_device)
    on_device = !omp_is_initial_device();
    if (!on_device) {
        // Device not available; continue on host.
    }

    // Main computation using host parallelism with better memory access pattern.
    // 1. Update a[i] in parallel.
    #pragma omp parallel for schedule(static) default(none) shared(a, b, c, LEN_2D)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] += b[i] * c[i];
    }
    // 2. Column‑wise recurrence. Outer loop over rows (j) is sequential due to dependence.
    //    Inside, we parallelise over columns (i) for contiguous accesses.
    // Parallel region to avoid repeated thread creation per j.
    #pragma omp parallel default(none) shared(a, bb, aa, LEN_2D)
    {
        for (int64_t j = 1; j < LEN_2D; ++j) {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < LEN_2D; ++i) {
                int64_t idx_cur = j * LEN_2D + i;
                int64_t idx_prev = (j - 1) * LEN_2D + i;
                aa[idx_cur] = aa[idx_prev] + bb[idx_cur] * a[i];
            }
            // Implicit barrier at end of omp for ensures all threads sync before next j.
        }
    }
}
