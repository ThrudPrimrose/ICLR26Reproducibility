/* Serial reference implementation of tsvc_2_s119_fp64.
 * This kernel updates a 2D array "aa" with a diagonal dependence:
 *   aa[i][j] = aa[i-1][j-1] + bb[i][j]
 * for i,j in [1, LEN_2D-1]. The loops are simple and fully serial, which
 * matches the reference baseline and guarantees correctness without OpenMP
 * overhead.
 */

#include <stdint.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    for (int64_t i = 1; i < LEN_2D; ++i) {
        for (int64_t j = 1; j < LEN_2D; ++j) {
            const int64_t idx_ij = i * LEN_2D + j;
            const int64_t idx_im1j = (i - 1) * LEN_2D + (j - 1);
            aa[idx_ij] = aa[idx_im1j] + bb[idx_ij];
        }
    }
}
