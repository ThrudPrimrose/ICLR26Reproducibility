#include <omp.h>
#include <stdint.h>
#include <stddef.h>

// Kernel implementation for TSVC tsvc_2_s255 (fp64 version).
// Expected signature as required by the harness:
//   void tsvc_2_s255_fp64(double *a, const double *b, int64_t LEN_1D,
//                        uint8_t *workspace, int64_t workspace_bytes);
// The kernel computes a[i] = (b[i] + b[(i-1) mod LEN_1D] + b[(i-2) mod LEN_1D]) * 0.333.
// The workspace arguments are unused but kept for API compatibility.
void tsvc_2_s255_fp64(double *a, const double *b, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    // Suppress unused parameter warnings
    (void)workspace;
    (void)workspace_bytes;
    const double c = 0.333;
    if (LEN_1D <= 0) return;
    if (LEN_1D == 1) {
        a[0] = (b[0] + b[0] + b[0]) * c;
        return;
    }
    if (LEN_1D == 2) {
        a[0] = (b[0] + b[1] + b[1]) * c;
        a[1] = (b[1] + b[0] + b[0]) * c;
        return;
    }
    // General case LEN_1D >= 3
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * c;
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * c;
    #pragma omp simd
    for (int64_t i = 2; i < LEN_1D; ++i) {
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * c;
    }
}
