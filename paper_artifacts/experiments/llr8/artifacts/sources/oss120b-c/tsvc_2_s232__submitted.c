#include <stdint.h>

void tsvc_2_s232_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    // Parallelize across rows (j). Each row's computation is a recurrence that depends only on previous column within the same row.
    // Use OpenMP to distribute rows across threads.
    #pragma omp parallel for schedule(static)
    for (int64_t j = 1; j < LEN_2D; ++j) {
        // Compute base pointer for row j in both arrays.
        double *restrict a_row = aa + j * LEN_2D;
        const double *restrict b_row = bb + j * LEN_2D;
        // The first column (i=0) is unchanged and used as seed.
        for (int64_t i = 1; i <= j; ++i) {
            double prev = a_row[i - 1];
            a_row[i] = prev * prev + b_row[i];
        }
    }
}

