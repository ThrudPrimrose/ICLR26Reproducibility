#include <stdint.h>
#include <omp.h>

// TSVC tsvc_2 kernel s1232
// Performs: aa[i,j] = bb[i,j] + cc[i,j] for i >= j*VLEN
// LEN_2D: dimension of square arrays
// VLEN: vector length controlling triangular region
// All arrays are assumed to be of type double and stored in row-major order.
// The kernel uses a loop transformation to make the inner loop contiguous for better cache locality and vectorization.
// Parallelism is applied over the outer loop (i), which is independent across rows.
void tsvc_2_s1232_fp64(double *aa, double *bb, double *cc, int64_t LEN_2D, int64_t VLEN, uint8_t *workspace, int64_t workspace_bytes) {
    // Parallelize over rows (i). Each row writes to a distinct set of columns up to floor(i / VLEN).
    #pragma omp parallel for schedule(static) default(none) shared(aa, bb, cc, LEN_2D, VLEN)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        // Maximum column j that satisfies i >= j*VLEN => j <= i / VLEN
        int64_t jmax = i / VLEN; // floor division
        // Vectorize the inner loop over j (contiguous memory stride = 1)
        #pragma omp simd
        for (int64_t j = 0; j <= jmax; ++j) {
            int64_t idx = i * LEN_2D + j;
            aa[idx] = bb[idx] + cc[idx];
        }
    }
}
