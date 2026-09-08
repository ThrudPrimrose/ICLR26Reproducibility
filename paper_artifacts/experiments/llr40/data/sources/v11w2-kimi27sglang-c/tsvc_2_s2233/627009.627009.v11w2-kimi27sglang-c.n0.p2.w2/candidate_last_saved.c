#include <stdint.h>
#include <omp.h>

#ifndef BLOCK
#define BLOCK 256
#endif

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t L = LEN_2D;

    /* aa[j,i] = aa[j-1,i] + cc[j,i]
       Independent over columns i; recurrence along rows j.
       One contiguous column block per thread, rows sequential, columns vectorised. */
    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 8; i0 < L; i0 += BLOCK) {
        int64_t i1 = i0 + BLOCK;
        if (i1 > L) i1 = L;
        for (int64_t j = 8; j < L; ++j) {
            #pragma omp simd
            for (int64_t i = i0; i < i1; ++i) {
                aa[j * L + i] = aa[(j - 1) * L + i] + cc[j * L + i];
            }
        }
    }

    /* bb[i,j] = bb[i-1,j] + cc[i,j]
       Independent over columns j; recurrence along rows i.
       One contiguous column block per thread, rows sequential, columns vectorised. */
    #pragma omp parallel for schedule(static)
    for (int64_t j0 = 8; j0 < L; j0 += BLOCK) {
        int64_t j1 = j0 + BLOCK;
        if (j1 > L) j1 = L;
        for (int64_t i = 8; i < L; ++i) {
            #pragma omp simd
            for (int64_t j = j0; j < j1; ++j) {
                bb[i * L + j] = bb[(i - 1) * L + j] + cc[i * L + j];
            }
        }
    }
}
