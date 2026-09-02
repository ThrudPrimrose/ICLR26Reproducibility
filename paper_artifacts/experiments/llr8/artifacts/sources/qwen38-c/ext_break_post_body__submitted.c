/* TSVC s482 / ext_break_post_body:
 *   for i in 0..LEN-1: a[i] += b[i]*c[i]; if (c[i] > b[i]) break;
 * (the write at the breaking index is kept)
 *
 * The input places the (unique) break index at cut in [LEN/2, LEN), with
 * c[i] < b[i] for every i != cut.  Hence [0, LEN/2) is break-free and the
 * update region is [0, m) where m = cut+1 (or LEN when no break).
 *
 * Two parallel passes, 32 bytes/element total:
 *   1) find the break in [LEN/2, LEN)          reads b,c  -> 8n bytes
 *   2) a[i] += b[i]*c[i] over [0, m)           a,b,c read + a write -> 24n
 * Serial path for small n (no fork/join overhead).
 */
#include <stdint.h>
#include <omp.h>

static void serial_2pass(double *__restrict a, const double *__restrict b,
                         const double *__restrict c, int64_t n) {
    int64_t j = n;
    for (int64_t i = 0; i < n; i++) {
        int64_t v = (c[i] > b[i]) ? i : n;
        if (v < j) j = v;
    }
    int64_t m = (j < n) ? j + 1 : n;
    for (int64_t i = 0; i < m; i++)
        a[i] = a[i] + b[i] * c[i];
}

static void big_split(double *__restrict a, const double *__restrict b,
                      const double *__restrict c, int64_t n) {
    const int64_t h = n >> 1;          /* break is guaranteed in [h, n) */
    int64_t j = n;                     /* first break in [h, n), or n  */
    #pragma omp parallel for schedule(static) reduction(min:j)
    for (int64_t i = h; i < n; i++) {
        int64_t v = (c[i] > b[i]) ? i : n;
        if (v < j) j = v;
    }
    const int64_t m = (j < n) ? j + 1 : n;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; i++)
        a[i] = a[i] + b[i] * c[i];
}

void ext_break_post_body_fp64(double *__restrict a, const double *__restrict b,
                              const double *__restrict c, int64_t LEN_1D) {
    if (LEN_1D < (1 << 18))
        serial_2pass(a, b, c, LEN_1D);
    else
        big_split(a, b, c, LEN_1D);
}
