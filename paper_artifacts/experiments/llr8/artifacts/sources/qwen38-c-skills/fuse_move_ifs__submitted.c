#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void fuse_move_ifs_fp64(
    double *restrict a,
    double *restrict b,
    const double *restrict cond,
    const double *restrict src,
    const int64_t K,
    const int64_t LEN_2D,
    unsigned char *restrict workspace,
    const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    const int64_t n = LEN_2D;
    if (n <= 0) return;

    if (K > 0) {
        /* Fused single pass over src: rows with cond[i]>0 also feed a.
           Row axis is independent (no dependence between rows) -> thread it;
           column axis is unit stride and disjoint (restrict) -> vectorize it. */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; i++) {
            const double *restrict s  = src + i * n;
            double *restrict ap = a + i * n;
            double *restrict bp = b + i * n;
            if (cond[i] > 0.0) {
                for (int64_t j = 0; j < n; j++) {
                    ap[j] = s[j] * 2.0;
                    bp[j] = s[j] + 1.0;
                }
            } else {
                for (int64_t j = 0; j < n; j++) {
                    bp[j] = s[j] + 1.0;
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; i++) {
            if (cond[i] > 0.0) {
                const double *restrict s  = src + i * n;
                double *restrict ap = a + i * n;
                for (int64_t j = 0; j < n; j++) {
                    ap[j] = s[j] * 2.0;
                }
            }
        }
    }
}
