/* Per-column GPU offload: each column is processed by a separate kernel launch.
 * The outer loop over columns (j) is executed on the host sequentially, while the
 * inner loop over rows (i) is offloaded and parallelized on the device. This
 * yields unit-stride memory accesses for aa and bb inside the kernel, which is
 * optimal for GPUs. Data is mapped once for the whole computation to avoid
 * repeated transfers.
 */

#include <stddef.h>
#include <stdint.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    #pragma omp target data map(tofrom: aa[0:LEN_2D*LEN_2D]) map(to: bb[0:LEN_2D*LEN_2D])
    {
        for (int64_t j = 1; j < LEN_2D; ++j) {
            #pragma omp target parallel for simd
            for (int64_t i = 0; i < LEN_2D; ++i) {
                aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i];
            }
        }
    }
}

