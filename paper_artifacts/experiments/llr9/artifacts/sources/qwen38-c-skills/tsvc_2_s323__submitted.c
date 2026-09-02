#include <stdint.h>

/* TSVC s323: coupled recurrence
 *   a[i] = b[i-1] + c[i]*d[i]
 *   b[i] = a[i]   + c[i]*e[i]
 * b is a left-fold (serial scan). Computed in exactly the reference order so the
 * result is bit-identical to the NumPy reference (no reassociation, no FMA).
 */
void tsvc_2_s323_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, int64_t LEN_1D) {
    for (int64_t i = 1; i < LEN_1D; ++i) {
        a[i] = b[i - 1] + c[i] * d[i];
        b[i] = a[i] + c[i] * e[i];
    }
}
