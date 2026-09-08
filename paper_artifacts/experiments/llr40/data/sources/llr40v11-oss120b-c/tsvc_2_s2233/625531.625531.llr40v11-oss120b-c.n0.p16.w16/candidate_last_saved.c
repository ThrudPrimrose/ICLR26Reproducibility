#include <stdint.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t start = 8;
    if (LEN_2D <= start) return; // nothing to do
    const int64_t count = LEN_2D - start;

    // Parallelize across columns of "aa" (each column is independent)
    #pragma omp parallel for schedule(static)
    for (int64_t i = start; i < LEN_2D; ++i) {
        const double *prev = aa + (start - 1) * LEN_2D + i; // aa[7,i]
        const double *cptr = cc + start * LEN_2D + i;      // cc[8,i]
        double *aptr = aa + start * LEN_2D + i;            // aa[8,i]
        for (int64_t k = 0; k < count; ++k) {
            *aptr = *prev + *cptr;
            prev += LEN_2D;
            cptr += LEN_2D;
            aptr += LEN_2D;
        }
    }

    // Parallelize across columns of "bb" (each column j is independent)
    #pragma omp parallel for schedule(static)
    for (int64_t j = start; j < LEN_2D; ++j) {
        const double *prev = bb + (start - 1) * LEN_2D + j; // bb[7,j]
        const double *cptr = cc + start * LEN_2D + j;      // cc[8,j]
        double *bptr = bb + start * LEN_2D + j;            // bb[8,j]
        for (int64_t k = 0; k < count; ++k) {
            *bptr = *prev + *cptr;
            prev += LEN_2D;
            cptr += LEN_2D;
            bptr += LEN_2D;
        }
    }
}
