#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
    // Edge case: if LEN_1D is zero, set result to 0 (undefined behavior otherwise)
    if (LEN_1D <= 0) {
        result[0] = 0.0;
        return;
    }

    // First pass: find the maximum absolute value.
    double maxv = 0.0;
    // Use OpenMP reduction to compute max safely.
    #pragma omp parallel for reduction(max:maxv) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = fabs(a[(size_t)i * (size_t)inc]);
        if (v > maxv) maxv = v; // reduction clause handles the combine
    }

    // Second pass: find the first index where the absolute value equals maxv.
    int64_t index = LEN_1D; // sentinel larger than any valid index
    #pragma omp parallel for reduction(min:index) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = fabs(a[(size_t)i * (size_t)inc]);
        // Equality is safe because maxv comes from the same set of values.
        if (v == maxv) {
            if (i < index) index = i; // reduction(min) will keep the smallest i
        }
    }
    // In the unlikely case maxv was not found (should not happen), fallback to 0.
    if (index == LEN_1D) index = 0;

    result[0] = maxv + (double)index;
}
