#include <stddef.h>
#include <stdint.h>

// TSVC_2 kernel s319 reference implementation.
// Parameters:
//   a, b, c, d, e: pointers to double arrays of length LEN_1D
//   LEN_1D: number of elements in each array.
// Behavior:
//   For each i in [0, LEN_1D):
//       a[i] = c[i] + d[i];
//       b[i] = c[i] + e[i];
//   After the loop, b[0] is set to the sum of all a[i] and b[i] values.
void tsvc_2_s319_fp64(double * restrict a,
                      double * restrict b,
                      const double * restrict c,
                      const double * restrict d,
                      const double * restrict e,
                      int64_t LEN_1D,
                      uint8_t *workspace,
                      int64_t workspace_bytes) {
    double sum_val = 0.0;
    // Vectorize the loop, accumulate sum in a local variable.
    #pragma omp parallel for simd schedule(static) reduction(+:sum_val)
    for (int i = 0; i < LEN_1D; ++i) {
        a[i] = c[i] + d[i];
        sum_val += a[i];
        b[i] = c[i] + e[i];
        sum_val += b[i];
    }
    // Store the reduction result.
    b[0] = sum_val;
    (void)workspace;
    (void)workspace_bytes;
}
