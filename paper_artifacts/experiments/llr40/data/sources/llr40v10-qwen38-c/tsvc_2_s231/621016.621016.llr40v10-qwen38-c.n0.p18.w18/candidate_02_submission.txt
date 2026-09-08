#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (N <= 1) return;
    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t i0 = (N * tid) / nt;
        const int64_t i1 = (N * (tid + 1)) / nt;
        if (i1 > i0) {
            for (int64_t j = 1; j < N; ++j) {
                double *a = aa + j * N;
                const double *ap = a - N;
                const double *b = bb + j * N;
                for (int64_t i = i0; i < i1; ++i)
                    a[i] = ap[i] + b[i];
            }
        }
    }
}