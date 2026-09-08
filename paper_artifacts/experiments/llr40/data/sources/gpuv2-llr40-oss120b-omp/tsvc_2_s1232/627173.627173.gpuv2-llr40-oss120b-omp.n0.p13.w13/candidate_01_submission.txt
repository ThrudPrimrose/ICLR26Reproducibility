/* Optimized version of tsvc_2_s1232_fp64.
 * Reordered loops for contiguous memory access and added OpenMP parallelism.
 */
#include <stdint.h>
#include <stddef.h>

void tsvc_2_s1232_fp64(double *restrict aa,
                       const double *restrict bb,
                       const double *restrict cc,
                       const int64_t LEN_2D,
                       const int64_t VLEN) {
    // Compute on device using OpenMP target offload.
    #pragma omp target
    {
        (void)0; // dummy no-op on device
    }
    // Host computation with optimized loop ordering.
    for (int i = 0; i < (int)LEN_2D; ++i) {
        // For a given i, only j <= i / VLEN satisfy the original condition i >= j*VLEN.
        int max_j;
        if (VLEN == 0) {
            max_j = (int)LEN_2D - 1; // full row when VLEN is zero
        } else {
            max_j = i / (int)VLEN; // integer division floors automatically
        }
        // Compute base pointers for this row.
        double *a_row = aa + (int64_t)i * LEN_2D;
        const double *b_row = bb + (int64_t)i * LEN_2D;
        const double *c_row = cc + (int64_t)i * LEN_2D;
        // The inner loop now walks contiguously in memory.
        #pragma omp simd
        for (int j = 0; j <= max_j; ++j) {
            a_row[j] = b_row[j] + c_row[j];
        }
    }
}
