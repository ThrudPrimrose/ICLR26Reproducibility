#include <stdint.h>

/* Optimized version of the TSVC tsvc_2 kernel "s235" for double precision.
   The reference implementation (see /shared/tasks/tsvc_2_s235/tsvc_2_s235_reference.c)
   updates the vector a and then computes a column‑wise prefix sum on the matrix aa.
   By separating the update of a from the nested loops and swapping the loop order we
   obtain unit‑stride access in the innermost loop, enabling the compiler to vectorize
   efficiently.  OpenMP SIMD directives are added to guide vectorization.
*/

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    /* Update a[i] = a[i] + b[i] * c[i] for all i. */
    #pragma omp simd
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] += b[i] * c[i];
    }

    /* Compute aa[j,i] = aa[j-1,i] + bb[j,i] * a[i] for j=1..LEN_2D-1.
       The loops are reordered so that the inner loop walks over i, which is
       contiguous in memory (column‑major storage). This yields a vectorizable
       stride‑1 access pattern. */
    for (int64_t j = 1; j < LEN_2D; ++j) {
        double *restrict aa_cur = aa + j * LEN_2D;
        double *restrict aa_prev = aa + (j - 1) * LEN_2D;
        const double *restrict bb_cur = bb + j * LEN_2D;

        #pragma omp simd
        for (int64_t i = 0; i < LEN_2D; ++i) {
            aa_cur[i] = aa_prev[i] + bb_cur[i] * a[i];
        }
    }
}
