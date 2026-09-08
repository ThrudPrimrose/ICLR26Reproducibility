/* Optimized version of TSVC tsvc_2_s1232 kernel.
   Reordered loops for better memory locality and added OpenMP parallelism.
   Original kernel performs:
     for (j=0; j<LEN_2D; ++j)
       for (i=j*VLEN; i<LEN_2D; ++i)
         aa[i*LEN_2D + j] = bb[i*LEN_2D + j] + cc[i*LEN_2D + j];
   This implementation swaps loops to iterate over rows `i` first, making inner accesses contiguous.
   The outer loop over `i` is parallelized with OpenMP.
*/

#include <stdint.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       const int64_t LEN_2D, const int64_t VLEN) {
    // Parallelize over rows (i). Each iteration works on independent output locations.
    #pragma omp parallel for schedule(static) // static schedule usually performs well for uniform work.
    for (int64_t i = 0; i < LEN_2D; ++i) {
        // Compute the maximum column index `j` that satisfies i >= j*VLEN.
        int64_t max_j = i / VLEN; // floor division.
        // The inner loop now walks contiguous memory (stride 1).
        // Use a base pointer to avoid repeated multiplication.
        double *aa_row = aa + i * LEN_2D;
        const double *bb_row = bb + i * LEN_2D;
        const double *cc_row = cc + i * LEN_2D;
        for (int64_t j = 0; j <= max_j; ++j) {
            aa_row[j] = bb_row[j] + cc_row[j];
        }
        // No work for j > max_j because condition i < j*VLEN would hold.
    }
}

