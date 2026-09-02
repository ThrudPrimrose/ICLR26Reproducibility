#include <stdint.h>
#include <omp.h>

/*
 * wf_north_west: a[i,j] = a[i,j] + a[i-1,j] + a[i,j-1] for i,j in [1, N).
 * Tile wavefront over anti-diagonals; inside a tile each row is the serial
 * recurrence c = c + (row[j] + up[j]) in exactly the reference's
 * left-to-right associativity -> bit-identical results (large N overflows,
 * any reassociation moves the Inf/NaN frontier). 4-cycle FP add chain per
 * element; rows are wide (DRAM-bound), so 3 rows ahead of the tile the row
 * data is prefetched into L2 and the tile-top up-row is prefetched too.
 */

#define TILE 128

static void compute_tile(double *a, int64_t N, int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
    for (int64_t i = r0; i < r1; i++) {
        double *row = a + i * N;
        const double *up = a + (i - 1) * N;
        int64_t half = (c1 - c0) / 2;
        if (i == r0) {
            __builtin_prefetch(up + c0, 0, 0);
            __builtin_prefetch(up + c0 + half, 0, 0);
        }
        if (i + 4 < N) {
            double *ahead = a + (i + 4) * N;
            __builtin_prefetch(ahead + c0, 0, 0);
            __builtin_prefetch(ahead + c0 + (c1 - c0) / 4, 0, 0);
            __builtin_prefetch(ahead + c0 + (c1 - c0) / 2, 0, 0);
            __builtin_prefetch(ahead + c0 + (3 * (c1 - c0)) / 4, 0, 0);
        }
        if (i + 3 < N) {
            double *ahead = a + (i + 3) * N;
            __builtin_prefetch(ahead + c0, 0, 0);
            __builtin_prefetch(ahead + c0 + half, 0, 0);
        }
        double c = row[c0 - 1];
        for (int64_t j = c0; j < c1; j++) {
            double b = row[j] + up[j];
            c = c + b;
            row[j] = c;
        }
    }
}

void wf_north_west_fp64(double *a, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    int64_t N = LEN_2D;
    if (N < 2) return;

    int64_t m = N - 1; /* rows/cols modified: 1..N-1 */
    int64_t R = (m + TILE - 1) / TILE;
    int64_t C = (m + TILE - 1) / TILE;

    #pragma omp parallel
    {
        for (int64_t d = 0; d < R + C - 1; d++) {
            int64_t bmin = d >= C - 1 ? d - (C - 1) : 0;
            int64_t bmax = d < R ? d + 1 : R;
            #pragma omp for schedule(static)
            for (int64_t bi = bmin; bi < bmax; bi++) {
                int64_t bj = d - bi;
                int64_t r0 = 1 + bi * TILE;
                int64_t r1 = r0 + TILE < N ? r0 + TILE : N;
                int64_t c0 = 1 + bj * TILE;
                int64_t c1 = c0 + TILE < N ? c0 + TILE : N;
                compute_tile(a, N, r0, r1, c0, c1);
            }
        }
    }
}
