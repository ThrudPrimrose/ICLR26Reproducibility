#include <stdint.h>
#include <omp.h>

/* TSVC tsvc_2 s252:
 *   t = 0.0
 *   for i in 0..N-1:
 *       s   = b[i] * c[i]
 *       a[i] = s + t
 *       t   = s
 *
 * `t` at iteration i is exactly s_{i-1}, a one-step shift that the loop
 * can recompute, so the carry is false:
 *   a[0] = b[0]*c[0]
 *   a[i] = b[i]*c[i] + b[i-1]*c[i-1]   (i >= 1)
 * Same products and add order as the reference (bitwise with FMA
 * contraction off) and completely free of carried state -> fully
 * parallel, one pass, unit stride.
 */
/* Reference does separate rounding of each product before the add;
 * disable FMA contraction so we match it bitwise.  The loop is memory
 * bound (3 arrays), so the extra add costs nothing. */
#pragma GCC optimize("fp-contract=off")
void tsvc_2_s252_fp64(double *a, double *b, double *c,
                      int64_t len_1d, uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    if (len_1d <= 0) return;

    double *restrict ar = a;
    const double *restrict br = b;
    const double *restrict cr = c;

    ar[0] = br[0] * cr[0];
    if (len_1d == 1) return;

    if (len_1d >= (1 << 14) && omp_get_max_threads() > 1) {
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 1; i < len_1d; i++)
            ar[i] = br[i] * cr[i] + br[i - 1] * cr[i - 1];
    } else {
        #pragma omp simd
        for (int64_t i = 1; i < len_1d; i++)
            ar[i] = br[i] * cr[i] + br[i - 1] * cr[i - 1];
    }
}
