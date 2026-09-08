#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    // Assume LEN_1D >= 2 (as per reference implementation)
    // Handle first two elements manually to respect wrap-around dependencies
    // using values from the end of b.
    // Use the same constant factor as reference.
    const double factor = 0.333; // matches reference constant
    if (LEN_1D <= 0) return;
    if (LEN_1D == 1) {
        // For LEN_1D == 1, the original code would read b[-1] and b[-2] which is undefined.
        // The benchmark never invokes this case, but we provide a safe fallback.
        a[0] = (b[0] + b[0] + b[0]) * factor;
        return;
    }
    // LEN_1D >= 2
    // i = 0
    double x0 = b[LEN_1D - 1];
    double y0 = b[LEN_1D - 2];
    a[0] = (b[0] + x0 + y0) * factor;
    // i = 1
    double x1 = b[0];
    double y1 = b[LEN_1D - 1];
    a[1] = (b[1] + x1 + y1) * factor;
    if (LEN_1D <= 2) return;
    // Main loop: independent computation using b[i], b[i-1], b[i-2]
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 2; i < LEN_1D; ++i) {
        a[i] = (b[i] + b[i-1] + b[i-2]) * factor;
    }
}
