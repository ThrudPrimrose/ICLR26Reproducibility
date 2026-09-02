#include <stdint.h>
#include <omp.h>
#include <math.h>

// Compute the maximum absolute value of elements from a strided array a, and the index of the max.
// a: input array of length at least (LEN_1D-1)*inc + 1, type double
// result: output array of length 1, stores max_abs + (double)index
// inc: stride between elements to consider (in elements)
// LEN_1D: number of elements to process
// The function runs in parallel using OpenMP.
void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, int64_t LEN_1D, int64_t inc, uint8_t *restrict workspace, int64_t workspace_bytes) {
    (void)workspace; (void)workspace_bytes;
    // Handle empty case (should not happen per spec, but guard anyway)
    if (LEN_1D <= 0) {
        result[0] = 0.0;
        return;
    }
    // Initialize with first element
    double maxv = fabs(a[0]);
    int64_t max_idx = 0;

    // Parallel reduction: each thread keeps a private best value and index
    double global_max = maxv;
    int64_t global_idx = max_idx;

    #pragma omp parallel
    {
        double thread_max = -INFINITY;
        int64_t thread_idx = -1;
        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double v = fabs(a[i * inc]);
            if (v > thread_max) {
                thread_max = v;
                thread_idx = i;
            }
        }
        // Combine thread results into global variables
        #pragma omp critical
        {
            if (thread_max > global_max || (thread_max == global_max && (global_idx == -1 || thread_idx < global_idx))) {
                global_max = thread_max;
                global_idx = thread_idx;
            }
        }
    }
    // If no thread updated (e.g., LEN_1D == 0), keep initial values
    result[0] = global_max + (double)global_idx;
}
