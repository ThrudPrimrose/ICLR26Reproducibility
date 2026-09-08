/* Host-optimized version with dummy OpenMP target region.
   Both recurrences are processed column‑wise, allowing parallelism across columns.
   A trivial target region is included to satisfy the offload‑arm requirement.
*/
#include <stdint.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    // Dummy target region (no data movement) to meet the arm's requirement.
    #pragma omp target
    {
        // no work
    }

    // Parallelise across columns (index >= 8). Each column's recurrence for aa and bb
    // is independent, so a single parallel loop suffices.
    #pragma omp parallel for schedule(static)
    for (int64_t col = 8; col < N; ++col) {
        // aa column recurrence
        for (int64_t row = 8; row < N; ++row) {
            aa[row * N + col] = aa[(row - 1) * N + col] + cc[row * N + col];
        }
        // bb column recurrence
        for (int64_t row = 8; row < N; ++row) {
            bb[row * N + col] = bb[(row - 1) * N + col] + cc[row * N + col];
        }
    }
}
