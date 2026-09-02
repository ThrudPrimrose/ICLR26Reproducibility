/* hpcagent_bench stub headers -- keep as given */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/*
 * wf_diff_skew:  for i in 1..N-1: for j in 0..N-2:
 *                  a[i,j] = a[i,j] + a[i-1,j] + a[i-1,j+1]
 *
 * Dependence vectors: (1,0) from a[i-1,j] and (1,-1) from a[i-1,j+1].
 * The second one preserves i+j, so the classic i+j wavefront degenerates
 * into a serial chain.  Grouping by i-j fixes it: a tile (I,J) reads
 * (a) row I*B-1 (first tile row) across columns J..J+1, and
 * (b) for every later tile row, the crossing column (J+1)*B of the row
 * above, which is written by the RIGHT neighbour in the same tile row.
 * Predecessors are therefore:
 *     (I-1, J)    diagonal (I-J) - 1   [case a, cols J*B..(J+1)*B-1]
 *     (I-1, J+1)  diagonal (I-J) - 2   [case a, col (J+1)*B]
 *     (I,   J+1)  diagonal (I-J) - 1   [case b, col (J+1)*B]
 * all strictly earlier block diagonals.  A wait for the J+1 tile is only
 * needed when (J+1)*B <= N-2; the last column N-1 is input, never
 * written, and when (J+1)*B >= N-1 there is no crossing column.
 *
 * Tiles of B x B on a T x T grid, streamed along the (I-J) anti-diagonals
 * in one persistent region; a tile spins (acquire) on its two predecessor
 * flags and publishes a release store after finishing.  The wait graph is
 * acyclic in diagonal order, so no deadlock.  Within a diagonal the tiles
 * are dealt round-robin, rotated by the diagonal number, so the ramp
 * diagonals (fewer than nt tiles) share work across ALL threads instead
 * of piling onto the low ids -- that imbalance cost ~15 ms of the 40 ms
 * wall before the fix.  T is capped at 96 (96 x 96 flags = 72 KB).
 */
void wf_diff_skew_fp64(double *a, int64_t LEN_2D, uint8_t *workspace,
                       int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    const int64_t N = LEN_2D;
    if (N < 2)
        return;

    const int64_t nthr = omp_get_max_threads();
    if (N < 128 || nthr < 2) {
        for (int64_t i = 1; i < N; i++) {
            double *__restrict ai = a + i * N;
            const double *__restrict aim1 = a + (i - 1) * N;
            for (int64_t j = 0; j < N - 1; j++)
                ai[j] = ai[j] + aim1[j] + aim1[j + 1];
        }
        return;
    }

    int64_t B = (N + 95) / 96;
    if (B < 64)
        B = 64;
    int64_t T = (N + B - 1) / B;
    if (T > 96) {
        B = (N + 95) / 96;
        T = (N + B - 1) / B;
    }
    static uint64_t flags[96 * 96];
    memset(flags, 0, (size_t)T * T * sizeof(uint64_t));
    const int64_t Smax = 2 * T - 1;

    #pragma omp parallel
    {
        const int64_t tid = omp_get_thread_num();
        const int64_t nt = omp_get_num_threads();
        for (int64_t s = 0; s < Smax; s++) {
            const int64_t d = s - (T - 1); /* I - J on this diagonal */
            int64_t J0 = (d < 0) ? -d : 0;
            int64_t J1 = (T - 1 - d < T - 1) ? T - 1 - d : T - 1;
            /* round-robin within the diagonal, rotated by s so the ramp
               diagonals (cnt < nt) share the work across ALL threads:
               thread t covers J = J0 + ((t+s) mod nt) + m*nt, and no two
               threads ever claim the same J on a diagonal. */
            for (int64_t J = J0 + ((tid + s) % nt); J <= J1; J += nt) {
                const int64_t I = J + d;
                const int64_t cross_ok = ((J + 1) * B <= N - 2);
                if (I > 0) {
                    const uint64_t *f1 = &flags[(I - 1) * T + J];
                    while (!__atomic_load_n(f1, __ATOMIC_ACQUIRE))
                        __builtin_ia32_pause();
                    if (cross_ok) {
                        const uint64_t *f2 = &flags[(I - 1) * T + (J + 1)];
                        while (!__atomic_load_n(f2, __ATOMIC_ACQUIRE))
                            __builtin_ia32_pause();
                    }
                }
                if (cross_ok) {
                    const uint64_t *f3 = &flags[I * T + (J + 1)];
                    while (!__atomic_load_n(f3, __ATOMIC_ACQUIRE))
                        __builtin_ia32_pause();
                }
                int64_t i0 = I * B;
                if (i0 < 1)
                    i0 = 1;
                int64_t i1 = (I + 1) * B;
                if (i1 > N)
                    i1 = N;
                int64_t j0 = J * B;
                int64_t j1 = (J + 1) * B;
                if (j1 > N - 1)
                    j1 = N - 1;
                for (int64_t i = i0; i < i1; i++) {
                    double *__restrict ai = a + i * N;
                    const double *__restrict aim1 = a + (i - 1) * N;
                    for (int64_t j = j0; j < j1; j++)
                        ai[j] = ai[j] + aim1[j] + aim1[j + 1];
                }
                __atomic_store_n(&flags[I * T + J], 1, __ATOMIC_RELEASE);
            }
        }
    }
}
