#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    if (LEN_2D <= 8) return;

    const int64_t n = LEN_2D - 8;

    /* For very small problems the OpenMP runtime overhead dominates. */
    if (n < 128) {
        for (int64_t j = 8; j < LEN_2D; ++j) {
            const int64_t row = j * LEN_2D;
            const int64_t prow = (j - 1) * LEN_2D;
            #pragma omp simd
            for (int64_t i = 8; i < LEN_2D; ++i) {
                aa[row + i] = aa[prow + i] + cc[row + i];
            }
        }
        for (int64_t j = 8; j < LEN_2D; ++j) {
            const int64_t row = j * LEN_2D;
            for (int64_t i = 8; i < LEN_2D; ++i) {
                bb[row + i] = bb[row + (i - 1)] + cc[row + i];
            }
        }
        return;
    }

    #pragma omp parallel
    {
        const int64_t nt = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();

        /* Phase A: aa[j,i] = aa[j-1,i] + cc[j,i].
           Recurrence is down rows (j), so j is serial inside each thread.
           Columns (i) are independent and unit-stride. */
        const int64_t i0 = 8 + (n * tid) / nt;
        const int64_t i1 = 8 + (n * (tid + 1)) / nt;

        for (int64_t j = 8; j < LEN_2D; ++j) {
            const int64_t row = j * LEN_2D;
            const int64_t prow = (j - 1) * LEN_2D;
            #pragma omp simd
            for (int64_t i = i0; i < i1; ++i) {
                aa[row + i] = aa[prow + i] + cc[row + i];
            }
        }

        /* Phase B: bb[j,i] = bb[j,i-1] + cc[j,i].
           Rows (j) are independent; recurrence is across columns (i). */
        const int64_t j0 = 8 + (n * tid) / nt;
        const int64_t j1 = 8 + (n * (tid + 1)) / nt;

        for (int64_t j = j0; j < j1; ++j) {
            const int64_t row = j * LEN_2D;
            for (int64_t i = 8; i < LEN_2D; ++i) {
                bb[row + i] = bb[row + (i - 1)] + cc[row + i];
            }
        }
    }
}
