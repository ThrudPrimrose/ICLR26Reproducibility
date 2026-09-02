#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

// Kernel function for TSVC kernel s318.
// Arguments:
//   a: pointer to input array of length LEN_1D * inc (or at least up to (LEN_1D-1)*inc)
//   result: pointer to output array of length 1 (stores the result)
//   inc: stride increment between elements to consider
//   LEN_1D: number of elements to process
// The kernel computes the maximum absolute value of the elements a[0], a[inc], a[2*inc], ...
// and the index (0-based) where that maximum occurs. The output is maxv + (float)index.

void tsvc_2_s318_fp64(double *a, double *result, int64_t LEN_1D, int64_t inc, uint8_t *workspace, int64_t workspace_bytes) {
    // Guard against zero length (should not happen in benchmark)
    (void)workspace; (void)workspace_bytes;
    if (LEN_1D <= 0) {
        result[0] = 0.0;
        return;
    }
    // Parallel reduction to find max absolute value and its first index.
    int nthreads = omp_get_max_threads();
    // Allocate thread-local arrays on the stack (VLA).
    double local_max[nthreads];
    int64_t local_idx[nthreads];
    // Initialize thread-local values.
    for (int t = 0; t < nthreads; ++t) {
        local_max[t] = -INFINITY;
        local_idx[t] = -1;
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double tmax = -INFINITY;
        int64_t tidx = -1;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int64_t k = i * inc;
            double v = fabs(a[k]);
            if (v > tmax) {
                tmax = v;
                tidx = i;
            }
            // If v == tmax keep earlier (lower) index.
        }
        local_max[tid] = tmax;
        local_idx[tid] = tidx;
    }
    // Reduce across threads to find global max and smallest index.
    double maxv = -INFINITY;
    int64_t maxidx = -1;
    for (int t = 0; t < nthreads; ++t) {
        if (local_max[t] > maxv) {
            maxv = local_max[t];
            maxidx = local_idx[t];
        } else if (local_max[t] == maxv && local_idx[t] >= 0 && (maxidx < 0 || local_idx[t] < maxidx)) {
            maxidx = local_idx[t];
        }
    }
    result[0] = maxv + (double)maxidx;
}

