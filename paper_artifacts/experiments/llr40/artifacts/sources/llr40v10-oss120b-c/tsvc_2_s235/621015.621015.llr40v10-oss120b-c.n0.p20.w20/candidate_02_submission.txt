#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    // Phase 1: update a[i] = a[i] + b[i] * c[i]
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] += b[i] * c[i];
    }

    // Phase 2: compute aa matrix column by column using contiguous row-major access.
    // Loop over rows (j) outermost to get contiguous access across columns (i).
    for (int64_t j = 1; j < LEN_2D; ++j) {
        double *aa_cur = aa + j * LEN_2D;
        double *aa_prev = aa + (j - 1) * LEN_2D;
        const double *bb_cur = bb + j * LEN_2D;
        // SIMD across columns i
        #pragma omp simd
        for (int64_t i = 0; i < LEN_2D; ++i) {
            aa_cur[i] = aa_prev[i] + bb_cur[i] * a[i];
        }
    }
}
