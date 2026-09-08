/* Optimized version of the TSVC tsvc_2_s119 kernel.
 * Original reference implementation:
 *   void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D)
 *   { for (int64_t i = 1; i < LEN_2D; ++i) {
 *       for (int64_t j = 1; j < LEN_2D; ++j) {
 *         aa[i*LEN_2D + j] = aa[(i-1)*LEN_2D + (j-1)] + bb[i*LEN_2D + j];
 *       }
 *     } }
 *
 * The kernel exhibits a wavefront (diagonal) dependence: aa[i][j] depends on aa[i-1][j-1].
 * To exploit parallelism, we reorder the iteration space into anti-diagonals (constant i+j).
 * Each anti-diagonal can be processed in parallel, while anti-diagonals themselves must be
 * performed sequentially.
 *
 * The implementation uses a persistent OpenMP team (pragma omp parallel) and a parallel for
 * over the inner loop for each anti-diagonal. The inner loop is also vectorizable: for a fixed
 * i, the accesses to aa and bb are unit-stride, so the compiler can emit SIMD instructions.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    #pragma omp parallel
    {
        for (int64_t k = 2; k <= 2 * N - 2; ++k) {
            int64_t i_start = k - (N - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = k - 1;
            if (i_end > N - 1) i_end = N - 1;
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                const int64_t j = k - i;
                const int64_t idx = i * N + j;
                const int64_t idx_prev = (i - 1) * N + (j - 1);
                aa[idx] = aa[idx_prev] + bb[idx];
            }
        }
    }
}
