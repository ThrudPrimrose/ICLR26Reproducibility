#include <stdint.h>
#include <omp.h>

/*
 * versioned_distance_update - compute a[i] = 0.75 * a[i-K] + b[i] * c[i]
 * for i = K .. LEN_1D-1.
 * The initial K elements of a are seed values and remain unchanged.
 * This implementation handles any K >= 1.
 * For K == 1 the recurrence is scalar and must be computed sequentially.
 * For K > 1 the computation consists of K independent chains; we parallelise across the chains.
 */
void versioned_distance_update_fp64(double *restrict a, const double *restrict b, const double *restrict c,
                               int64_t K_val, int64_t LEN_1D_val, uint8_t *workspace, int64_t workspace_bytes) {
    // Use the GPU for modest problem sizes; otherwise fall back to host execution.
    if (LEN_1D_val <= 20000000) {
        // Offload to the device.
        #pragma omp target map(to: b[0:LEN_1D_val], c[0:LEN_1D_val]) map(tofrom: a[0:LEN_1D_val])
        #pragma omp teams distribute parallel for schedule(static)
        for (int64_t offset = 0; offset < K_val; ++offset) {
            for (int64_t i = offset + K_val; i < LEN_1D_val; ++i) {
                a[i] = 0.75 * a[i - K_val] + b[i] * c[i];
            }
        }
    } else {
        // Host execution for large inputs.
        #pragma omp parallel for schedule(static)
        for (int64_t offset = 0; offset < K_val; ++offset) {
            for (int64_t i = offset + K_val; i < LEN_1D_val; ++i) {
                a[i] = 0.75 * a[i - K_val] + b[i] * c[i];
            }
        }
    }
}
