#include <math.h>
#include <stdint.h>
#include <omp.h>

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
    if (LEN_1D <= 0) {
        result[0] = 0.0;
        return;
    }
    // Compute maximum absolute value with parallel reduction
    double maxv = fabs(a[0]);
    #pragma omp parallel for reduction(max:maxv) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = fabs(a[i * inc]);
        if (v > maxv) {
            maxv = v;
        }
    }
    // Find the first index where this maximum occurs
    int64_t idx = LEN_1D; // sentinel
    #pragma omp parallel for reduction(min:idx) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = fabs(a[i * inc]);
        if (v >= maxv) {
            idx = i;
        }
    }
    if (idx == LEN_1D) idx = 0;
    result[0] = maxv + (double)idx;
}
