#include <stdint.h>
#include <omp.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb,
                       const double *restrict cc, const int64_t LEN_2D) {

    const int64_t n = LEN_2D - 8;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        /* Phase 1: each thread scans its own columns of aa along j. */
        {
            const int64_t i_start = 8 + (int64_t)tid * n / nthreads;
            const int64_t i_end = 8 + (int64_t)(tid + 1) * n / nthreads;

            for (int64_t j = 8; j < LEN_2D; ++j) {
                #pragma omp simd
                for (int64_t i = i_start; i < i_end; ++i) {
                    aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
                }
            }
        }

        /* Phase 2: each thread scans its own rows of bb along i. */
        {
            const int64_t j_start = 8 + (int64_t)tid * n / nthreads;
            const int64_t j_end = 8 + (int64_t)(tid + 1) * n / nthreads;

            for (int64_t i = 8; i < LEN_2D; ++i) {
                #pragma omp simd
                for (int64_t j = j_start; j < j_end; ++j) {
                    bb[i * LEN_2D + j] = bb[(i - 1) * LEN_2D + j] + cc[i * LEN_2D + j];
                }
            }
        }
    }
}
