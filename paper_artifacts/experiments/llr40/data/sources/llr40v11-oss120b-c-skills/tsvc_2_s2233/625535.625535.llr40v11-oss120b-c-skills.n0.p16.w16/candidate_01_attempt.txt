/* Optimized implementation of tsvc_2_s2233_fp64.
   Original reference (see /shared/tasks/tsvc_2_s2233/tsvc_2_s2233_reference.c) performs two
   recurrences:
   1) A column‑wise recurrence on array aa, independent across the outer index i.
   2) A row‑wise recurrence on array bb, independent across the inner index j.

   The recurrence in (1) prevents vectorising the inner loop, but the outer i‑loop can be
   parallelised because each column is independent. The recurrence in (2) prevents parallelising
   the outer i‑loop, but the inner j‑loop is embarrassingly parallel and can be SIMD‑vectorised.

   This implementation therefore:
     * Parallelises the outer i‑loop of the aa‑update with OpenMP.
     * SIMD‑vectorises the inner j‑loop of the bb‑update with an OpenMP simd directive.
   The loops preserve the original ordering and numerical result, while exploiting multi‑core
   parallelism and vectorisation where safe.
*/

#include <stdint.h>
#include <omp.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    /* Phase 1: vertical sweep on aa (recurrence along j).
       Independent across i, so we parallelise the outer i‑loop.
       The inner j‑loop has a forward dependence (aa[j] depends on aa[j-1]),
       therefore it remains scalar.
    */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 8; i < LEN_2D; ++i) {
        for (int64_t j = 8; j < LEN_2D; ++j) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
        }
    }

    /* Phase 2: horizontal sweep on bb (recurrence along i).
       The outer i‑loop carries a dependence, so it stays serial.
       The inner j‑loop is free of dependencies and can be vectorised.
    */
    for (int64_t i = 8; i < LEN_2D; ++i) {
        #pragma omp simd
        for (int64_t j = 8; j < LEN_2D; ++j) {
            bb[i * LEN_2D + j] = bb[(i - 1) * LEN_2D + j] + cc[i * LEN_2D + j];
        }
    }
}

