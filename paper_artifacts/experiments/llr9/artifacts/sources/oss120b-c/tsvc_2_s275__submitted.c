#include <stdlib.h>
#include <stdint.h>

// Entry point expected by the benchmark harness.
// Signature: void <kernel>_fp64(double *aa, double *bb, double *cc,
//                               int64_t LEN_2D, uint8_t *workspace,
//                               int64_t workspace_bytes);
// The function performs the operation defined by the reference implementation.
void tsvc_2_s275_fp64(double *aa, double *bb, double *cc,
                      int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    int n = (int)LEN_2D;
    // Allocate a temporary mask to store the condition aa[0,i] > 0.0 for each column i.
    // This avoids repeatedly evaluating the condition inside the inner loop.
    char *mask = (char *)malloc((size_t)n * sizeof(char));
    if (!mask) {
        // Allocation failed; cannot proceed safely.
        return;
    }
    for (int i = 0; i < n; ++i) {
        mask[i] = (aa[i] > 0.0) ? 1 : 0; // aa[0,i] corresponds to element at row 0, column i.
    }
    // Perform the column‑wise prefix sum for rows 1..n-1.
    for (int j = 1; j < n; ++j) {
        double *aa_row = aa + j * n;
        double *aa_prev = aa + (j - 1) * n;
        double *bb_row = bb + j * n;
        double *cc_row = cc + j * n;
        for (int i = 0; i < n; ++i) {
            if (mask[i]) {
                aa_row[i] = aa_prev[i] + bb_row[i] * cc_row[i];
            }
        }
    }
    free(mask);
    // Workspace is unused for this kernel.
}
