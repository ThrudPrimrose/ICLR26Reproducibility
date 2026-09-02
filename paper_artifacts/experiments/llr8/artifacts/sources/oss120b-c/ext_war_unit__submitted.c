#include <omp.h>
#include <stdint.h>
#include <stdlib.h>

void ext_war_unit_fp64(double *restrict a, double *restrict b, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    // Compute a[i] = a[i+1] + b[i] for i = 0 .. LEN_1D-2.
    // The last element a[LEN_1D-1] remains unchanged.
    int64_t n = LEN_1D - 1;
    if (n <= 0) return;
    if (workspace && workspace_bytes >= n * sizeof(double)) {
        double *tmp = (double *)workspace;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            tmp[i] = a[i+1];
        }
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            a[i] = tmp[i] + b[i];
        }
    } else {
        // Allocate temporary buffer if no workspace provided.
        double *tmp = (double *)malloc(n * sizeof(double));
        if (tmp) {
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                tmp[i] = a[i+1];
            }
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                a[i] = tmp[i] + b[i];
            }
            free(tmp);
        } else {
            // Fallback scalar loop if allocation fails.
            #pragma GCC ivdep
            for (int64_t i = 0; i < n; ++i) {
                a[i] = a[i+1] + b[i];
            }
        }
    }
}
