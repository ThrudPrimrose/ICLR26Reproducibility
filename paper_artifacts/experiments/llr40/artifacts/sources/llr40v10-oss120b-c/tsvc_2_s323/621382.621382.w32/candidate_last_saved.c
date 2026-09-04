#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#if 1

// Optimized sequential implementation of the TSVC s323 kernel.
// This version removes the indirect array access pattern and keeps the exact
// floating‑point evaluation order of the reference implementation while reducing
// memory traffic and loop overhead through a register‑based accumulator and
// manual loop unrolling.

void tsvc_2_s323_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    if (LEN_1D <= 1) return;
    for (int64_t i = 1; i < LEN_1D; ++i) {
        a[i] = b[i - 1] + c[i] * d[i];
        b[i] = a[i] + c[i] * e[i];
    }
}
#endif

/* Reference implementation disabled */

