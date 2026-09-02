#include <stdint.h>
#include <omp.h>

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    int nthreads = omp_get_max_threads();
    // Allocate per-thread partial sums on the stack.
    double partials[nthreads];
    for (int i = 0; i < nthreads; ++i) {
        partials[i] = 0.0;
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double local = 0.0;
        #pragma omp for schedule(static)
        for (int64_t i = 1; i < LEN_1D; i += 2) {
            local += a[i];
        }
        partials[tid] = local;
    }
    double acc = 0.0;
    for (int i = 0; i < nthreads; ++i) {
        acc += partials[i];
    }
    out[0] = acc;
}
