#include <stdint.h>
#include <omp.h>

/*
 * Optimized implementation of the TSVC tsvc_2_s2233 microkernel (fp64 version).
 * The reference implementation updates two 2D arrays "aa" and "bb" based on a recurrence.
 *
 *   aa[j, i] = aa[j-1, i] + cc[j, i]   // column‑major layout, recurrence across j
 *   bb[i, j] = bb[i-1, j] + cc[i, j]   // row‑major layout, recurrence across i
 *
 * The dependencies allow parallelism across the dimension that does NOT carry the recurrence.
 *   * For the "aa" update, each column (fixed i) is independent -> we can parallelise the outer i loop.
 *   * For the "bb" update, each row (fixed i) depends on the previous row -> we cannot parallelise i,
 *     but the inner j loop is free of dependencies and can be vectorised.
 *
 * The implementation therefore:
 *   - Uses an OpenMP parallel for over i for the aa loop.
 *   - Uses an OpenMP simd directive on the j loop of the bb update for vectorisation.
 *   - Keeps the original loop bounds (starting at 8) to match the reference.
 */

void tsvc_2_s2233_fp64(double *restrict aa,
                       double *restrict bb,
                       const double *restrict cc,
                       const int64_t LEN_2D)
{
    // Update aa: compute prefix sum across columns for each row i.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 8; i < LEN_2D; ++i) {
        double prev = aa[(8 - 1) * LEN_2D + i]; // aa[7*LEN_2D + i]
        for (int64_t j = 8; j < LEN_2D; ++j) {
            prev = prev + cc[j * LEN_2D + i];
            aa[j * LEN_2D + i] = prev;
        }
    }

    // Update bb: compute prefix sum across rows for each column j.
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < LEN_2D; ++j) {
        double prev = bb[(8 - 1) * LEN_2D + j]; // bb[7*LEN_2D + j]
        for (int64_t i = 8; i < LEN_2D; ++i) {
            prev = prev + cc[i * LEN_2D + j];
            bb[i * LEN_2D + j] = prev;
        }
    }
}

