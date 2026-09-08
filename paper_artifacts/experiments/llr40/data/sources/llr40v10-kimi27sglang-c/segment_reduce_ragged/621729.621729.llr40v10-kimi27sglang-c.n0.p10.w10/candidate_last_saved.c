#include <stdint.h>

void segment_reduce_ragged_fp64(double *out_, int64_t *row_ptr_, double *val_, double *w_, int64_t NSEG, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    double *__restrict__ out = __builtin_assume_aligned(out_, 64);
    int64_t *__restrict__ row_ptr = __builtin_assume_aligned(row_ptr_, 64);
    double *__restrict__ val = __builtin_assume_aligned(val_, 64);
    double *__restrict__ w = __builtin_assume_aligned(w_, 64);
    #pragma omp parallel for schedule(guided) if(NSEG > 8192)
    for (int64_t s = 0; s < NSEG; s++) {
        int64_t start = row_ptr[s];
        int64_t end = row_ptr[s + 1];
        double acc = 0.0;
        #pragma omp simd reduction(+:acc) aligned(val, w: 64)
        for (int64_t e = start; e < end; e++) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
