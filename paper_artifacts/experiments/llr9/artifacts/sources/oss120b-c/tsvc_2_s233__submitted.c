/*
 * TSVC_2 kernel s233 reference implementation in C.
 *
 * The Python reference (tsvc_2_s233_numpy.py) defines the kernel as:
 *
 *   def s233(aa, bb, cc, LEN_2D):
 *       for i in range(8, LEN_2D):
 *           for j in range(8, LEN_2D):
 *               aa[j, i] = aa[j - 1, i] + cc[j, i]
 *           for j in range(8, LEN_2D):
 *               bb[j, i] = bb[j, i - 1] + cc[j, i]
 *
 * In C we store the 2-D arrays in row-major order (the default for NumPy)
 * so element (j,i) is at index j*LEN_2D + i.
 *
 * The kernel is exposed with the name expected by the benchmark harness:
 *   void tsvc_2_s233(double *aa, double *bb, const double *cc, int LEN_2D)
 *
 * This version implements the reference exactly, with plain loops.
 */

#include <stddef.h>
#include <stdint.h>

void tsvc_2_s233_fp64(double *aa, double *bb, const double *cc, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    // Vertical prefix sum on aa (column-wise). Parallel over columns.
#pragma omp parallel for schedule(static)
for (int i = 8; i < LEN_2D; ++i) {
    double *aa_ptr = aa + (int64_t)8 * LEN_2D + i;      // aa[8,i]
    const double *cc_ptr = cc + (int64_t)8 * LEN_2D + i; // cc[8,i]
    double *aa_prev = aa_ptr - LEN_2D;                 // aa[7,i]
    for (int j = 8; j < LEN_2D; ++j) {
        *aa_ptr = *aa_prev + *cc_ptr;
        aa_prev = aa_ptr;
        aa_ptr += LEN_2D;
        cc_ptr += LEN_2D;
    }
}
// Horizontal prefix sum on bb (row-wise). Must be sequential in i to respect dependencies.
for (int i = 8; i < LEN_2D; ++i) {
    double *bb_cur = bb + (int64_t)8 * LEN_2D + i;          // bb[8,i]
    double *bb_left = bb + (int64_t)8 * LEN_2D + (i - 1);   // bb[8,i-1]
    const double *cc_cur = cc + (int64_t)8 * LEN_2D + i;    // cc[8,i]
    for (int j = 8; j < LEN_2D; ++j) {
        *bb_cur = *bb_left + *cc_cur;
        bb_cur += LEN_2D;
        bb_left += LEN_2D;
        cc_cur += LEN_2D;
    }
}
}

