/* scan_affine_decay kernel - simple serial implementation */
/* Computes y[i] = c[i] * y[i-1] + x[i] for i = 1..LEN_1D-1 */

#include <stdint.h>

/* Expected signature (as required by the harness):
   void scan_affine_decay_fp64(const double *restrict c,
                               const double *restrict x,
                               double *restrict y,
                               int64_t LEN_1D,
                               uint8_t *workspace,
                               int64_t workspace_bytes);
   The workspace arguments are ignored as this kernel does not need temporary storage.
*/

void scan_affine_decay_fp64(const double *restrict c,
                            const double *restrict x,
                            double *restrict y,
                            const int64_t LEN_1D,
                            uint8_t *workspace,
                            const int64_t workspace_bytes) {
    if (LEN_1D <= 0) return;
    /* The initializer already sets y[0] = x[0]; set it again for safety */
    y[0] = x[0];
    for (int64_t i = 1; i < LEN_1D; ++i) {
        y[i] = c[i] * y[i - 1] + x[i];
    }
    /* workspace is unused */
}
