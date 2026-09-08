#include <stdint.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                        const double *restrict e, const int64_t LEN_1D) {
    // Guard against trivial case
    if (LEN_1D <= 1) return;

    // Number of elements to process (i = 1 .. LEN_1D-1)
    int64_t n = LEN_1D - 1;
    // Maximum number of threads we may use
    int max_threads = omp_get_max_threads();
    // Compute block size for static schedule (same as OpenMP default)
    int64_t chunk = (n + max_threads - 1) / max_threads; // ceil division

    // Allocate per‑thread sum array (block sums of inc = c*d + c*e)
    double *block_sum = (double *)aligned_alloc(64, max_threads * sizeof(double));
    // Initialise to zero (in case some threads are idle)
    for (int i = 0; i < max_threads; ++i) block_sum[i] = 0.0;

    // First pass: each thread computes the sum of its block's increments.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = 1 + tid * chunk;
        int64_t end   = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        double sum = 0.0;
        for (int64_t i = start; i < end; ++i) {
            // inc[i] = c[i] * d[i] + c[i] * e[i]
            sum += c[i] * d[i] + c[i] * e[i];
        }
        block_sum[tid] = sum;
    }

    // Compute exclusive prefix sums of block_sum to obtain the starting b value for each block.
    double *offset = (double *)aligned_alloc(64, max_threads * sizeof(double));
    double cur = b[0]; // b[0] is the initial value used by the sequential algorithm
    for (int i = 0; i < max_threads; ++i) {
        offset[i] = cur;
        cur += block_sum[i]; // round as double addition, matching sequential rounding across block boundaries
    }

    // Second pass: each thread recomputes a[i] and b[i] using its block offset.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = 1 + tid * chunk;
        int64_t end   = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        // If this thread has no work, simply return.
        if (start >= end) return;
        double b_prev = offset[tid]; // b value before the first element of this block
        for (int64_t i = start; i < end; ++i) {
            double a_i = b_prev + c[i] * d[i];
            double b_i = a_i + c[i] * e[i];
            a[i] = a_i;
            b[i] = b_i;
            b_prev = b_i; // propagate to next iteration in the block
        }
    }

    // Clean up temporary buffers
    free(block_sum);
    free(offset);
}

