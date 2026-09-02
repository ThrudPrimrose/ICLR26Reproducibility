#include <stddef.h>
#include <stdint.h>
#include <omp.h>

// Optimized TSVC s2710 kernel (fp64).
// Signature required by the harness: void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c, const double *restrict d, const double *restrict e, const double *restrict x, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes);

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c, const double *restrict d, const double *restrict e, const double *restrict x, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D <= 0) return;
    bool len_gt_10 = LEN_1D > 10;
    bool x0_gt0 = x[0] > 0.0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > b[i]) {
            a[i] = a[i] + b[i] * d[i];
            c[i] = len_gt_10 ? (c[i] + d[i] * d[i]) : (d[i] * e[i] + 1.0);
        } else {
            b[i] = a[i] + e[i] * e[i];
            c[i] = x0_gt0 ? (a[i] + d[i] * d[i]) : (c[i] + e[i] * e[i]);
        }
    }
}
