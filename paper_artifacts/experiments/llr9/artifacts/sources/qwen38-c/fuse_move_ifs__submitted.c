/* TSVC tsvc_2_5 fuse_move_ifs — optimized C.
 *
 * ABI (binding order): (a, b, cond, src, K, LEN_2D, workspace, workspace_bytes)
 *
 *   for i: if cond[i] > 0: a[i,:] = 2*src[i,:]
 *   if K > 0: b[i,:] = src[i,:] + 1   (all rows)
 *
 * Both statements read src; fused into one pass so src is read once.
 * Rows are independent -> threaded over i; inner loops are unit-stride
 * and alias-free (restrict) -> auto-vectorized.
 */
#include <stdint.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b,
                        const double *restrict cond, const double *restrict src,
                        int64_t K, int64_t LEN_2D,
                        uint8_t *ws, int64_t ws_bytes)
{
    (void)ws; (void)ws_bytes;
    if (K > 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const double *s = src + i * LEN_2D;
            double *av = a + i * LEN_2D;
            double *bv = b + i * LEN_2D;
            if (cond[i] > 0.0) {
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    av[j] = s[j] * 2.0;
                    bv[j] = s[j] + 1.0;
                }
            } else {
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    bv[j] = s[j] + 1.0;
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const double *s = src + i * LEN_2D;
            double *av = a + i * LEN_2D;
            if (cond[i] > 0.0) {
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    av[j] = s[j] * 2.0;
                }
            }
        }
    }
}
