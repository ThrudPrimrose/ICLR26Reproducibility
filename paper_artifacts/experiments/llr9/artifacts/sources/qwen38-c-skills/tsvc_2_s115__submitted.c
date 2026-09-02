#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef BLOCK
#define BLOCK 32
#endif

void tsvc_2_s115_fp64(
    double *restrict a,
    const double *restrict aa,
    const int64_t LEN_2D,
    uint8_t *restrict workspace,
    const int64_t workspace_size)
{
    const int64_t N = LEN_2D;
    const int64_t B = BLOCK;
    (void)workspace;
    (void)workspace_size;

    if (N < 2 * B) {
        for (int64_t j = 0; j < N; ++j) {
            const double aj = a[j];
            const double *restrict row = aa + j * N;
            for (int64_t i = j + 1; i < N; ++i) a[i] -= aj * row[i];
        }
        return;
    }

    const int64_t m = (N + B - 1) / B;
#pragma omp parallel
    {
        for (int64_t k = 0; k < m; ++k) {
            const int64_t kstart = k * B;
            const int64_t Bend   = (k + 1) * B < N ? (k + 1) * B : N;
            const int64_t Bk     = Bend - kstart;
            /* within-block forward substitution (sequential in p) */
#pragma omp single
            for (int64_t p = 0; p < Bk; ++p) {
                double s = 0.0;
                for (int64_t q = 0; q < p; ++q)
                    s += a[kstart + q] * aa[(kstart + q) * N + kstart + p];
                a[kstart + p] -= s;
            }
            /* trailing update: for each block i>k, a[i*] -= M[i,k] * a[k*] */
#pragma omp for schedule(static)
            for (int64_t i = k + 1; i < m; ++i) {
                const int64_t istart = i * B;
                const int64_t iend   = (i + 1) * B < N ? (i + 1) * B : N;
                const int64_t Bik    = iend - istart;
                double acc[BLOCK];
                for (int64_t p = 0; p < Bik; ++p) acc[p] = 0.0;
                for (int64_t q = 0; q < B; ++q) {
                    const double xk = a[kstart + q];
                    const double *restrict row = aa + (kstart + q) * N + istart;
                    for (int64_t p = 0; p < Bik; ++p)
                        acc[p] += xk * row[p];
                }
                for (int64_t p = 0; p < Bik; ++p)
                    a[istart + p] -= acc[p];
            }
        }
    }
}
