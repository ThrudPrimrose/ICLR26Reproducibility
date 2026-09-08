#include <stdint.h>

/* tsvc_2_s323 : coupled recurrence
 *   a[i] = b[i-1] + c[i]*d[i]
 *   b[i] = a[i]   + c[i]*e[i]
 * This is a prefix-scan with a 2-op (2-FADD) dependency chain per index, so it is
 * irreducibly serial. The data is an ill-conditioned random walk, so ANY re-association
 * of the summation order exceeds the graded fp64 tolerance (rtol=1e-9); the rounding
 * must match the serial left-to-right order exactly. We therefore keep the exact two-add
 * form and let the compiler carry the running b in a register.
 */
void tsvc_2_s323_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D,
                      uint8_t *restrict workspace, const int64_t workspace_size) {
  (void)workspace; (void)workspace_size;
  for (int64_t i = 1; i < LEN_1D; ++i) {
    a[i] = b[i - 1] + c[i] * d[i];
    b[i] = a[i] + c[i] * e[i];
  }
}
