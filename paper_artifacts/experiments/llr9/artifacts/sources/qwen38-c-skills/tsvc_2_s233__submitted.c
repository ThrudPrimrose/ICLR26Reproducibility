#include <stdint.h>
#include <omp.h>
#include <string.h>

/* TSVC tsvc_2_s233:
 *   for i in 8..N-1:  aa[j,i] = aa[j-1,i] + cc[j,i]   (j scan per column, parallel over i)
 *   for i in 8..N-1:  bb[j,i] = bb[j,i-1] + cc[j,i]   (i scan per row,     parallel over j)
 * The two statements touch disjoint arrays -> fissioned into two phases.
 */
typedef double v8d __attribute__((vector_size(64)));

void tsvc_2_s233_fp64(double* aa, double* bb, double* cc, int64_t LEN_2D,
                                    uint8_t* workspace, int64_t workspace_bytes)
{
    const int64_t N = LEN_2D;
    (void)workspace; (void)workspace_bytes;
    if (N <= 8) return;

    const int64_t nb = (N - 8) / 8; /* full 8-column blocks; tail handled serially */

    #pragma omp parallel
    {
        /* Phase 1: aa. One 8-wide block per iteration; 8 independent j-scans,
         * one SIMD add per row. Full 64B lines per row per block. */
        #pragma omp for schedule(static) nowait
        for (int64_t b = 0; b < nb; b++) {
            const int64_t i0 = 8 + b * 8;
            v8d acc;
            for (int64_t v = 0; v < 8; v++) acc[v] = aa[7 * N + i0 + v];
            for (int64_t j = 8; j < N; j++) {
                const int64_t row = j * N + i0;
                v8d c;
                for (int64_t v = 0; v < 8; v++) c[v] = cc[row + v];
                acc = acc + c;
                for (int64_t v = 0; v < 8; v++) aa[row + v] = acc[v];
            }
        }
        /* Phase 2: bb. Unit-stride i-scan per row; sequential chain in a register. */
        #pragma omp for schedule(static)
        for (int64_t j = 8; j < N; j++) {
            double acc = bb[j * N + 7];
            const int64_t row = j * N;
            for (int64_t i = 8; i < N; i++) {
                acc += cc[row + i];
                bb[row + i] = acc;
            }
        }
    }
    /* Tail columns of aa: at most 7 columns, serial. */
    for (int64_t i = 8 + 8 * nb; i < N; i++) {
        double acc = aa[7 * N + i];
        for (int64_t j = 8; j < N; j++) {
            acc += cc[j * N + i];
            aa[j * N + i] = acc;
        }
    }
}
