#include <omp.h>

// Triangular north+west wavefront over j >= i; the parallel front is the i+j anti-diagonal.
// a is a LEN_2D x LEN_2D matrix stored in row-major order.
// The function updates a in-place: a[i][j] = a[i][j] + a[i-1][j] + a[i][j-1] for i>=1 and j>=i.
// It is parallelized across anti-diagonals using OpenMP.

void wf_triangular_fp64(double *restrict a, long long LEN_2D, unsigned char *workspace, long long workspace_bytes) {
    // anti-diagonal sum ranges from 2 (i=1,j=1) to 2*LEN_2D - 2 (i=LEN_2D-1,j=LEN_2D-1)
    #pragma omp parallel
    {
        for (int sum = 2; sum <= 2 * LEN_2D - 2; ++sum) {
            // i must satisfy: 1 <= i <= LEN_2D-1, i <= sum - i (j >= i) and j = sum - i < LEN_2D
            // Compute lower and upper bounds for i.
            int i_start = sum - (LEN_2D - 1);
            if (i_start < 1) i_start = 1;
            int i_end = sum / 2; // floor division ensures i <= j
            if (i_end > LEN_2D - 1) i_end = LEN_2D - 1;
            #pragma omp for schedule(static)
            for (int i = i_start; i <= i_end; ++i) {
                int j = sum - i;
                // Compute linear indices.
                int idx = i * LEN_2D + j;
                int idx_up = (i - 1) * LEN_2D + j;
                int idx_left = i * LEN_2D + (j - 1);
                a[idx] = a[idx] + a[idx_up] + a[idx_left];
            }
        }
    }
}

// Optional naive reference implementation for testing.
void wf_triangular_naive(double *a, int LEN_2D) {
    for (int i = 1; i < LEN_2D; ++i) {
        for (int j = i; j < LEN_2D; ++j) {
            int idx = i * LEN_2D + j;
            int idx_up = (i - 1) * LEN_2D + j;
            int idx_left = i * LEN_2D + (j - 1);
            a[idx] = a[idx] + a[idx_up] + a[idx_left];
        }
    }
}

