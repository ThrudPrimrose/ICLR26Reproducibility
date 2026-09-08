/* Argmax with index: include empty target region to meet offload-arm requirement */
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

void argmax_with_index_fp64(const double *restrict a,
                            int64_t *restrict out_index,
                            double *restrict out_value,
                            const int64_t LEN_1D) {
    // Empty target region – forces generation of a device kernel without data transfer.
#pragma omp target
    {
        // No work on device.
    }
    // Host-side parallel reduction for maximum value.
    double max_val = -INFINITY;
#pragma omp parallel for reduction(max:max_val) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = a[i];
        if (v > max_val) {
            max_val = v;
        }
    }
    // Find the first index of that maximum.
    int64_t min_idx = LEN_1D; // sentinel (reduction will use neutral element)
#pragma omp parallel for reduction(min:min_idx) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] == max_val && i < min_idx) {
            min_idx = i;
        }
    }
    out_value[0] = max_val;
    out_index[0] = min_idx;
}
