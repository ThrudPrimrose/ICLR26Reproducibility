// scan_affine_decay kernel implementation in C++.
//
// This kernel mirrors the NumPy reference implementation located at
// /shared/tasks/scan_affine_decay/scan_affine_decay_numpy.py.
//
// The recurrence is:
//   y[0] = x[0]
//   y[i] = c[i] * y[i-1] + x[i]   for i = 1 .. LEN_1D-1
//
// The function is called by the benchmark harness with the "_fp64"
// suffix.  The argument order follows the NumPy signature:
//   (y, c, x, LEN_1D) where "y" is the output array that also serves as the
//   seed for the recurrence, "c" and "x" are read‑only inputs, and LEN_1D is the
//   1‑D length of all arrays.
//
// A pure serial implementation is correct but does not exploit the available
// parallelism.  The dependence is linear (first‑order affine).  By partitioning
// the iteration space into contiguous blocks we can compute, for each block, an
// affine transformation that maps the value before the block (y_before) to the
// value after the block.  Those block‑level transforms are independent and can be
// obtained in parallel.  A cheap sequential prefix scan of the block transforms
// yields the correct starting value for each block, after which a second
// sequential pass fills the block's portion of the output.  This yields O(N)
// work with a small O(num_threads) sequential component and scales well on many
// cores.
//
// The algorithm is numerically identical to the serial version because each
// element is processed in the original order once the correct block start
// value is known.  No re‑ordering of arithmetic occurs, so floating‑point
// rounding matches the reference exactly.
//
// Note: The kernel must handle the edge case LEN_1D == 0 gracefully.
//
// Compilation flags (provided by the judge) include -O3, -march=native,
// -fopenmp, -fno-math-errno, -fno-trapping-math, -fno-signed-zeros,
// -fstrict-aliasing, -Wall, -Wextra.

#include <cstdint>
#include <vector>
#include <omp.h>

extern "C" void scan_affine_decay_fp64(const double* __restrict__ c,
                                       const double* __restrict__ x,
                                       double* __restrict__ y,
                                       const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    // Seed the recurrence: y[0] = x[0]
    y[0] = x[0];
    if (LEN_1D == 1) return;

    const int64_t n = LEN_1D;
    // Determine number of threads to use.
    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;
    omp_set_dynamic(0);

    // Work range is indices [1, n-1]
    const int64_t work_len = n - 1;
    // Block size (ceil division)
    const int64_t block_len = (work_len + num_threads - 1) / num_threads;

    std::vector<double> block_A(num_threads);
    std::vector<double> block_B(num_threads);
    std::vector<int64_t> block_start(num_threads);
    std::vector<int64_t> block_end(num_threads);

    // First pass: compute per‑block affine transform (A, B)
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        int64_t start = 1 + tid * block_len;
        int64_t end = start + block_len - 1;
        if (end >= n) end = n - 1;
        if (start > n - 1) {
            // No work for this thread.
            block_A[tid] = 1.0;
            block_B[tid] = 0.0;
            block_start[tid] = n;
            block_end[tid] = n - 1;
        } else {
            block_start[tid] = start;
            block_end[tid] = end;
            double A = c[start];
            double B = x[start];
            for (int64_t i = start + 1; i <= end; ++i) {
                A = c[i] * A;
                B = c[i] * B + x[i];
            }
            block_A[tid] = A;
            block_B[tid] = B;
        }
    }

    // Prefix scan over blocks to determine the starting y value for each block.
    std::vector<double> block_y_start(num_threads);
    double prev_y = y[0];
    for (int t = 0; t < num_threads; ++t) {
        if (block_start[t] > block_end[t]) {
            // Empty block, keep the current value.
            block_y_start[t] = prev_y;
            continue;
        }
        block_y_start[t] = prev_y;
        // Apply the block transform to obtain y after this block.
        prev_y = block_A[t] * prev_y + block_B[t];
    }

    // Second pass: fill y values within each block using its starting y.
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        int64_t start = block_start[tid];
        int64_t end = block_end[tid];
        if (start > end) { /* no work */ } else {
        double y_val = block_y_start[tid]; // y before this block (y[start-1])
        for (int64_t i = start; i <= end; ++i) {
            double yi = c[i] * y_val + x[i];
            y[i] = yi;
            y_val = yi;
        }
        }
    }
}
