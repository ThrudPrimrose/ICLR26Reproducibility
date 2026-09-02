#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <omp.h>

// TSVC_2 kernel s13110 implementation (double precision).
// Finds the maximum value in a LEN_2D x LEN_2D matrix 'aa' and its indices,
// then stores max + row_index + col_index into bb[0].
// The workspace arguments are unused but required by the ABI.

void tsvc_2_s13110_fp64(double *restrict aa, double *restrict bb, int64_t LEN_2D,
                       uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;       // Unused, required by the ABI.
    (void)workspace_bytes; // Unused, required by the ABI.
    if (LEN_2D <= 0) return;

    // Initialise with the first element.
    double maxv = aa[0];
    int64_t max_i = 0;
    int64_t max_j = 0;

    // Parallel reduction to find the maximum and its location.
    #pragma omp parallel
    {
        double local_max = -DBL_MAX;
        int64_t local_i = -1;
        int64_t local_j = -1;
        #pragma omp for collapse(2) schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            for (int64_t j = 0; j < LEN_2D; ++j) {
                double v = aa[i * LEN_2D + j];
                if (v > local_max) {
                    local_max = v;
                    local_i = i;
                    local_j = j;
                }
            }
        }
        #pragma omp critical
        {
            if (local_max > maxv) {
                maxv = local_max;
                max_i = local_i;
                max_j = local_j;
            }
        }
    }

    double chksum = maxv + (double)max_i + (double)max_j;
    bb[0] = chksum;
}
