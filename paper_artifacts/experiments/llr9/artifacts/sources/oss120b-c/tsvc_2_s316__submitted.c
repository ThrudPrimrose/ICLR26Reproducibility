#include <float.h>
#include <stdint.h>
#include <omp.h>

// Kernel for TSVC 2, kernel s316, double-precision (fp64).
// Expected entry point: void tsvc_2_s316_fp64(double *a, double *result,
//                        int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes);
// The workspace arguments are unused for this kernel.

void tsvc_2_s316_fp64(double *a, double *result, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    // Find the minimum element in a[0..LEN_1D-1] and store it in result[0].
    // Use parallel reduction to improve performance on large inputs.
    double min_val = DBL_MAX;
    #pragma omp parallel for reduction(min:min_val) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = a[i];
        if (v < min_val) {
            min_val = v;
        }
    }
    result[0] = min_val;
    // workspace and workspace_bytes are unused; they are present for ABI compatibility.
    (void)workspace;
    (void)workspace_bytes;
}

