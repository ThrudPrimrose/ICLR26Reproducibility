#include <stdint.h>
#include <omp.h>
#include <math.h>

// Compute the maximum value in array a and its index.
// a: input array of length LEN_1D (type double)
// out_value: output array of length 1, stores the maximum value
// out_index: output array of length 1, stores the index of the maximum value
// LEN_1D: length of a
// The function runs sequentially.
void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value, int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_bytes) {
        // Parallel reduction to find the maximum value and its first index.
    double global_max = -INFINITY;
    int64_t global_idx = -1;
    #pragma omp parallel
    {
        double local_max = -INFINITY;
        int64_t local_idx = -1;
        #pragma omp for nowait schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double val = a[i];
            if (val > local_max) {
                local_max = val;
                local_idx = i;
            }
        }
        #pragma omp critical
        {
            if (local_max > global_max) {
                global_max = local_max;
                global_idx = local_idx;
            } else if (local_max == global_max && local_idx < global_idx) {
                // Tie: pick the smaller index.
                global_idx = local_idx;
            }
        }
    }
    out_value[0] = global_max;
    out_index[0] = global_idx;
}
