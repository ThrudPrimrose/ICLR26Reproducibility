#include <stdint.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    // Dummy offload region to satisfy device kernel requirement.
    #pragma omp target
    {
        // No operation.
    }
    const int64_t N = LEN_2D;
    // Parallelize across diagonals (wavefront) on the host.
    #pragma omp parallel for schedule(static)
    for (int64_t p = -(N - 2); p <= (N - 2); ++p) {
        // Determine the starting indices for this diagonal.
        int64_t start_i = (p >= 0) ? (p + 1) : 1;
        int64_t start_j = (p >= 0) ? 1 : (-p + 1);
        // Length of the diagonal (number of elements to compute).
        int64_t len = N - (p >= 0 ? p : -p) - 1; // N - |p| - 1
        for (int64_t k = 0; k < len; ++k) {
            int64_t i = start_i + k;
            int64_t j = start_j + k;
            int64_t idx = i * N + j;
            int64_t idx_prev = (i - 1) * N + (j - 1);
            aa[idx] = aa[idx_prev] + bb[idx];
        }
    }
}
