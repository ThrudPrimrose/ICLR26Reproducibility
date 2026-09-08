#include <math.h>
#include <stdint.h>
#include <omp.h>

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
    if (LEN_1D <= 0) {
        result[0] = 0.0;
        return;
    }
    // First pass: compute maximum absolute value
    double maxv = fabs(a[0]);
    #pragma omp parallel for reduction(max:maxv) schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double v = fabs(a[i * inc]);
        if (v > maxv) {
            maxv = v;
        }
    }
    // Second pass: find the smallest index where the max occurs
    int64_t idx = LEN_1D; // sentinel large value
    #pragma omp parallel for reduction(min:idx) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = fabs(a[i * inc]);
        if (v == maxv && i < idx) {
            idx = i;
        }
    }
    // idx should be set; if not, fallback to 0
    if (idx == LEN_1D) idx = 0;
    result[0] = maxv + (double)idx;
}
