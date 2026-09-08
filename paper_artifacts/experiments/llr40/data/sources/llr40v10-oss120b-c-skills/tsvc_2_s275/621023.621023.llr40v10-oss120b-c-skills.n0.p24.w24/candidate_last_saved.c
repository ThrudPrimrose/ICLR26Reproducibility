/* Parallel implementation of tsvc_2_s275_fp64 with reduced memory traffic.
   The recurrence aa[j,i] = aa[j-1,i] + bb[j,i] * cc[j,i] can be rewritten using an accumulator
   to avoid reading aa[(j-1),i] from memory each iteration. This reduces memory loads and
   improves bandwidth usage while keeping the outer column loop parallel.
   Reference implementation: /shared/tasks/tsvc_2_s275/tsvc_2_s275_reference.c
*/

#include <stdint.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa,
                     const double *restrict bb,
                     const double *restrict cc,
                     const int64_t LEN_2D) {
    // Parallelise over columns (i). Each column is independent.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        double seed = aa[i];
        if (seed > 0.0) {
            double acc = seed;
            // Pointers to the start of row 1 for this column.
            double *aa_ptr = aa + (LEN_2D + i);
            const double *bb_ptr = bb + (LEN_2D + i);
            const double *cc_ptr = cc + (LEN_2D + i);
            // Iterate over remaining rows.
            for (int64_t j = 1; j < LEN_2D; ++j) {
                acc += (*bb_ptr) * (*cc_ptr);
                *aa_ptr = acc;
                aa_ptr += LEN_2D;
                bb_ptr += LEN_2D;
                cc_ptr += LEN_2D;
            }
        }
    }
}
