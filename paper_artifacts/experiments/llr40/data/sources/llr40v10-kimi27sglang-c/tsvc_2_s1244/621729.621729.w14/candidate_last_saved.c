#include <stdint.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
    const int64_t niter = LEN_1D - 1;
    if (niter <= 0) return;

    int nt = omp_get_max_threads();
    double boundary[nt];

    // Pre-pass: each thread saves the a[end] boundary value that its block will need
    // for the last iteration, before any thread writes it.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk = (niter + nthreads - 1) / nthreads;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > niter) end = niter;
        if (start < end) {
            boundary[tid] = a[end];
        }
    }

    // Main pass: each block runs the original loop sequentially, using the saved
    // boundary value for the last iteration to avoid cross-block dependencies.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk = (niter + nthreads - 1) / nthreads;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > niter) end = niter;

        // Vectorize all but the last iteration of the block
        #pragma omp simd
        for (int64_t i = start; i < end - 1; i++) {
            a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
            d[i] = a[i] + a[i + 1];
        }
        // Last iteration uses the saved boundary value
        if (start < end) {
            int64_t i = end - 1;
            a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
            d[i] = a[i] + boundary[tid];
        }
    }
}
