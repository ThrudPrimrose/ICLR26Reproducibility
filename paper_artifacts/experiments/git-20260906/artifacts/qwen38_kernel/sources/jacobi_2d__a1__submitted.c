#include <stdint.h>
#include <omp.h>

/* 2-D Jacobi 5-point stencil, PolyBench-style ping-pong between A and B.
 * Only the interior [1, N-2]^2 is ever written; boundary rows/cols keep
 * their initial values in both arrays, so both sweeps may read them freely.
 *
 * Strategy:
 *  - tiny grids: plain serial code (parallelism overhead would dominate),
 *  - larger grids: ONE persistent OpenMP team for the whole TSTEPS loop.
 *    Each thread owns a contiguous band of rows for every sweep, so
 *    the band's halo rows are re-read from cache/L3 instead of DRAM.
 */

static void sweep_row_A_to_B(const double *a0, const double *a1,
                             const double *a2, double *b1, int64_t n) {
    for (int64_t j = 1; j < n; ++j)
        b1[j] = 0.2 * (a1[j] + a0[j] + a2[j] + a1[j - 1] + a1[j + 1]);
}

static void sweep_row_B_to_A(const double *b0, const double *b1,
                             const double *b2, double *a1, int64_t n) {
    for (int64_t j = 1; j < n; ++j)
        a1[j] = 0.2 * (b1[j] + b0[j] + b2[j] + b1[j - 1] + b1[j + 1]);
}

static void jacobi_serial(double *restrict A, double *restrict B,
                          int64_t N, int64_t TSTEPS) {
    const int64_t ni = N - 2;
    for (int64_t t = 0; t < TSTEPS; ++t) {
        for (int64_t i = 1; i <= ni; ++i)
            sweep_row_A_to_B(A + (i - 1) * N, A + i * N, A + (i + 1) * N,
                             B + i * N, ni);
        for (int64_t i = 1; i <= ni; ++i)
            sweep_row_B_to_A(B + (i - 1) * N, B + i * N, B + (i + 1) * N,
                             A + i * N, ni);
    }
}

static void jacobi_team(double *restrict A, double *restrict B,
                        int64_t N, int64_t TSTEPS) {
    const int64_t ni = N - 2;
    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t r0 = (ni * tid) / nt;        /* first owned 0-based row of A-index i-1 */
        const int64_t r1 = (ni * (tid + 1)) / nt;  /* one past last owned */
        for (int64_t t = 0; t < TSTEPS; ++t) {
            #pragma omp barrier
            for (int64_t i = r0 + 1; i <= r1; ++i)
                sweep_row_A_to_B(A + (i - 1) * N, A + i * N, A + (i + 1) * N,
                                 B + i * N, ni);
            #pragma omp barrier
            for (int64_t i = r0 + 1; i <= r1; ++i)
                sweep_row_B_to_A(B + (i - 1) * N, B + i * N, B + (i + 1) * N,
                                 A + i * N, ni);
        }
    }
}

void jacobi_2d_fp64(double *restrict A, double *restrict B,
                    int64_t N, int64_t TSTEPS) {
    if (N <= 2 || TSTEPS <= 0) return;
    if (N < 1024) {
        jacobi_serial(A, B, N, TSTEPS);
    } else {
        jacobi_team(A, B, N, TSTEPS);
    }
}
