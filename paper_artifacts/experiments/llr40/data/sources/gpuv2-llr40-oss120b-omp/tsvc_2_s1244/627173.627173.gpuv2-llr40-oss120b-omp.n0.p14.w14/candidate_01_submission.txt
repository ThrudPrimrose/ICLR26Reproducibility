#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
    if (LEN_1D <= 1) return; // nothing to do
    // Allocate temporary copy of original a
    double *a_orig = (double*)malloc((size_t)LEN_1D * sizeof(double));
    if (!a_orig) {
        // fallback to scalar computation if allocation fails
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            double ai = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
            a[i] = ai;
            d[i] = ai + a[i + 1];
        }
        return;
    }
    // Copy original a values
    memcpy(a_orig, a, (size_t)LEN_1D * sizeof(double));

    #pragma omp target map(tofrom: a[0:LEN_1D]) map(to: b[0:LEN_1D], c[0:LEN_1D], a_orig[0:LEN_1D]) map(tofrom: d[0:LEN_1D])
    {
        #pragma omp teams distribute parallel for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            double ai = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
            a[i] = ai;
            d[i] = ai + a_orig[i + 1];
        }
    }
    free(a_orig);
}
