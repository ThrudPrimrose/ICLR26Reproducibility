#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* TSVC s235 (fp64):
 *   a[i]    = a[i] + b[i]*c[i]
 *   aa[j,i] = aa[j-1,i] + bb[j,i]*a[i]   (j=1..N-1)
 *
 * The dependence runs down a column (fixed i); columns are independent.
 * Each thread takes a 128-column segment and carries the running sums in
 * `acc` (L1/register-resident), streaming the bb rows: the write to aa[j]
 * never feeds a load, so the row stores stay off the critical path.
 * Per-element order is mul-then-add, exactly the numpy reference.
 */

#define SEG 128

static inline void scan_full(const double *restrict a, double *restrict aa,
                             const double *restrict bb, const int64_t n,
                             const int64_t seg0)
{
    double acc[SEG];
    for (int64_t k = 0; k < SEG; k++) acc[k] = aa[seg0 + k];
    for (int64_t j = 1; j < n; j++) {
        const double *restrict brow = bb + j * n + seg0;
        double *restrict crow = aa + j * n + seg0;
        for (int64_t k = 0; k < SEG; k++)
            acc[k] = acc[k] + brow[k] * a[seg0 + k];
        for (int64_t k = 0; k < SEG; k++)
            crow[k] = acc[k];
    }
}

static inline void scan_tail(double *restrict a, double *restrict aa,
                             const double *restrict bb, const int64_t n,
                             const int64_t seg0, const int64_t cnt)
{
    for (int64_t j = 1; j < n; j++) {
        const double *restrict brow = bb + j * n + seg0;
        const double *restrict prev = aa + (j - 1) * n + seg0;
        double *restrict crow = aa + j * n + seg0;
        for (int64_t k = 0; k < cnt; k++)
            crow[k] = prev[k] + brow[k] * a[seg0 + k];
    }
}

void tsvc_2_s235_fp64(
    double *restrict a,
    double *restrict aa,
    const double *restrict b,
    const double *restrict bb,
    const double *restrict c,
    const int64_t LEN_2D,
    uint8_t *restrict workspace,
    const int64_t workspace_size)
{
    const int64_t n = LEN_2D;
    (void)workspace;
    (void)workspace_size;
    if (n <= 0) return;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        a[i] = a[i] + b[i] * c[i];
    }

    const int64_t nseg_full = n / SEG;
    const int64_t nseg = (n + SEG - 1) / SEG;

    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < nseg; s++) {
        const int64_t seg0 = s * SEG;
        if (s < nseg_full)
            scan_full(a, aa, bb, n, seg0);
        else
            scan_tail(a, aa, bb, n, seg0, n - seg0);
    }
}
