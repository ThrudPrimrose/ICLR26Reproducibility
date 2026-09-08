#include <stdint.h>
#include <omp.h>

#ifndef BLOCK
#define BLOCK 432
#endif

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const int64_t stride = (int64_t)nthreads * BLOCK;
        for (int64_t ii = (int64_t)tid * BLOCK; ii < LEN_2D; ii += stride) {
            const int64_t iend = (ii + BLOCK < LEN_2D) ? ii + BLOCK : LEN_2D;
            for (int64_t j = 1; j < LEN_2D; ++j) {
                for (int64_t i = ii; i < iend; ++i) {
                    aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i];
                }
            }
        }
    }
}
