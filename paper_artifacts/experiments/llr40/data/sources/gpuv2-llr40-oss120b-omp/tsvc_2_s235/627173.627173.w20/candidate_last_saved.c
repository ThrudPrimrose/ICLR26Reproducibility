#include <stdint.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    // Offload the entire computation to the device using a single combined target teams distribute parallel for.
    // Map all arrays: a and aa are both read/write; others are read-only.
    #pragma omp target teams distribute parallel for \
        map(tofrom: a[0:LEN_2D]) \
        map(to: b[0:LEN_2D], c[0:LEN_2D], bb[0:LEN_2D*LEN_2D]) \
        map(tofrom: aa[0:LEN_2D*LEN_2D])
    for (int64_t i = 0; i < LEN_2D; ++i) {
        // Update a[i]
        a[i] += b[i] * c[i];
        // Compute aa for this column across rows j = 1..LEN_2D-1
        for (int64_t j = 1; j < LEN_2D; ++j) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i] * a[i];
        }
    }
}
