/* TSVC_2 kernel vtvtv reference implementation in C.
 *
 * The kernel computes element-wise triple multiplication:
 *   a[i] = a[i] * b[i] * c[i]
 * for i in [0, LEN_1D). This matches the NumPy reference.
 *
 * Expected signature (C linkage):
 *   void tsvc_2_vtvtv_fp64(double *a, const double *b, const double *c,
 *                          int64_t LEN_1D, uint8_t *workspace,
 *                          int64_t workspace_bytes);
 *
 * The workspace arguments are unused for this kernel.
 */

#include <omp.h>
#include <stdint.h>
#include <stddef.h>

void tsvc_2_vtvtv_fp64(double *restrict a,
                       const double *restrict b,
                       const double *restrict c,
                       int64_t LEN_1D,
                       uint8_t *restrict workspace,
                       int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    // Hint alignment for the compiler.
    a = (double *) __builtin_assume_aligned(a, 64);
    b = (const double *) __builtin_assume_aligned(b, 64);
    c = (const double *) __builtin_assume_aligned(c, 64);
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }
}
