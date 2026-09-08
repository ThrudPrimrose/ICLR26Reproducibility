#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    if (LEN_2D <= 256) {
        // Small size: use contiguous access with vectorization
        for (int64_t j = 1; j < LEN_2D; ++j) {
            double *restrict aa_cur = aa + j * LEN_2D;
            double *restrict aa_prev = aa + (j - 1) * LEN_2D;
            const double *restrict bb_cur = bb + j * LEN_2D;
            #pragma omp simd
            for (int64_t i = 0; i < LEN_2D; ++i) {
                aa_cur[i] = aa_prev[i] + bb_cur[i];
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            double prev = aa[i];
            double *aa_ptr = aa + i;
            const double *bb_ptr = bb + i;
            int64_t j = 1;
            // Unrolled loop, processing 4 elements per iteration
            for (; j + 3 < LEN_2D; j += 4) {
                // iteration 1
                aa_ptr += LEN_2D;
                bb_ptr += LEN_2D;
                double cur1 = prev + *bb_ptr;
                *aa_ptr = cur1;
                // iteration 2
                aa_ptr += LEN_2D;
                bb_ptr += LEN_2D;
                double cur2 = cur1 + *bb_ptr;
                *aa_ptr = cur2;
                // iteration 3
                aa_ptr += LEN_2D;
                bb_ptr += LEN_2D;
                double cur3 = cur2 + *bb_ptr;
                *aa_ptr = cur3;
                // iteration 4
                aa_ptr += LEN_2D;
                bb_ptr += LEN_2D;
                double cur4 = cur3 + *bb_ptr;
                *aa_ptr = cur4;
                prev = cur4;
            }
            // Remainder loop
            for (; j < LEN_2D; ++j) {
                aa_ptr += LEN_2D;
                bb_ptr += LEN_2D;
                double cur = prev + *bb_ptr;
                *aa_ptr = cur;
                prev = cur;
            }
        }
    }
}
