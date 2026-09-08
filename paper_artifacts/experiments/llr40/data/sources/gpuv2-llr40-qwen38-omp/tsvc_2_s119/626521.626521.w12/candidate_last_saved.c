/* tsvc_2_s119: aa[i][j] = aa[i-1][j-1] + bb[i][j], i,j = 1..LEN_2D-1 (in-place on aa).
 *
 * Data-flow facts: row i depends only on row i-1; within a row the columns are
 * independent. The first row (row 1) is a pure elementwise op that offloads to the
 * device as a real OpenMP target region. Rows 2..LEN_2D-1 run on the host where the
 * 24-core slot's DRAM (the data is host-resident and the host<->device link is far
 * slower than node-local DRAM): each thread owns a contiguous column chunk of every
 * row, so its previous-row chunk is an L1-resident stream, stores are full-line
 * vector stores, and the only cross-thread edge (one element, the chunk's left
 * neighbour) is published with an acquire/release flag pair.
 */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D)
{
    const int64_t L = LEN_2D;
    if (L < 2) return;

    /* Row 1 on the device: aa[1][j] = aa[0][j-1] + bb[1][j]. Row 0 and column 0 of
     * aa are inputs, never written: the two aa rows cross as ONE contiguous tofrom
     * section (row 0 comes back bit-identical) and only row 1 of bb crosses in.
     * Non-contiguous sections of one pointer are rejected by this runtime, so the
     * slices must be contiguous. */
    #pragma omp target map(tofrom: aa[0:2 * L]) map(to: bb[L + 1 : L - 1])
    {
        #pragma omp parallel for
        for (int64_t j = 1; j < L; ++j)
            aa[L + j] = aa[j - 1] + bb[L + j];
    }

    /* Rows 2..L-1 on the host. */
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    const int64_t W = (L - 1 + nthreads - 1) / nthreads;

    /* published[c] = last row fully written by thread c (row 1 is pre-existing). */
    unsigned published[64];
    for (int c = 0; c < nthreads; ++c) published[c] = 1;

    #pragma omp parallel num_threads(nthreads)
    {
        const int c = omp_get_thread_num();
        const int64_t jstart = 1 + (int64_t)c * W;
        int64_t jend = jstart + W;
        if (jend > L) jend = L;
        const int64_t n = jend - jstart; /* columns owned: [jstart, jstart+n) */
        for (int64_t i = 2; i < L; ++i) {
            if (n > 0) {
                double *restrict a0 = aa + i * L + jstart;
                const double *restrict a1 = aa + (i - 1) * L + jstart - 1;
                const double *restrict b0 = bb + i * L + jstart;
                for (int64_t j = 1; j < n; ++j) /* columns jstart+1..: own previous chunk */
                    a0[j] = a1[j] + b0[j];
                if (c > 0) /* column jstart needs thread c-1's column jstart-1, row i-1 */
                    while (__atomic_load_n(&published[c - 1], __ATOMIC_ACQUIRE) < (unsigned)(i - 1))
                        __builtin_ia32_pause();
                a0[0] = a1[0] + b0[0];
            }
            __atomic_store_n(&published[c], (unsigned)i, __ATOMIC_RELEASE);
        }
    }
}
