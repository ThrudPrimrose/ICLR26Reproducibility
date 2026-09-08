#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    int64_t n = LEN_2D;
    if (n <= 0) return;

    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();

        // Chunk size rounded up so that each thread gets a multiple of 8 doubles (one cache line).
        int64_t block = ((n + nt - 1) / nt + 7) & ~((int64_t)7);
        int64_t i0 = (int64_t)tid * block;
        int64_t i1 = i0 + block;
        if (i1 > n) i1 = n;

        if (i0 < i1) {
            // First update a[i] for this thread's columns.
            #pragma omp simd
            for (int64_t i = i0; i < i1; ++i) {
                a[i] += b[i] * c[i];
            }

            // Then scan down the rows, keeping columns contiguous in the inner loop.
            for (int64_t j = 1; j < n; ++j) {
                const double *restrict prev = aa + (j - 1) * n;
                const double *restrict brow = bb + j * n;
                double *restrict cur = aa + j * n;
                #pragma omp simd
                for (int64_t i = i0; i < i1; ++i) {
                    cur[i] = prev[i] + brow[i] * a[i];
                }
            }
        }
    }
}
