/*
 * TSVC-style kernel:  for i in 0..LEN_1D-2:  a[i] = a[i+1] + b[i]
 *
 * The in-place loop carries a write-read anti-dependence chain (iteration i
 * writes a[i], iteration i-1 reads a[i]), but the RESULT of every iteration
 * depends only on the ORIGINAL a[i+1] and b[i] -- a[i+1] is always read before
 * the later iteration that overwrites it.  So per element:
 *      a[i]_new = a[i+1]_orig + b[i]
 * and the problem is purely about ordering, not a recurrence.
 *
 * Split the iteration range [0, n-1] (n = LEN_1D-1) into blocks.  Inside a
 * block we process ascending with an 8-wide "load-ahead" chunk: the chunk at
 * [i, i+7] loads a[i+1..i+8] BEFORE storing a[i..i+7].  Consecutive chunks
 * never overlap (chunk 2's first read is lo2+1 > chunk 1's last write), so the
 * whole block is in-place correct, and each chunk is vector-load / add /
 * vector-store.  The ONLY cross-block read is the last iteration of block k
 * reading the first element of block k+1; that value is prefetched into
 * boundary[k] before any block starts writing.  All blocks are independent ->
 * OpenMP-parallel.  Total traffic: read a, read b, write a (minimal).
 */
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#define CHUNK 8
#define BLOCK (CHUNK * 1024)          /* 8192 iterations per block */
#define PAR_THRESHOLD (1 << 20)       /* use threads above this many iterations */

static void block_work(double *a, double *b, int64_t lo, int64_t hi, double boundary)
{
    int64_t i = lo;
    /* chunks whose iterations fit in [lo, hi-1]; iteration hi uses boundary */
    for (; i + CHUNK <= hi; i += CHUNK) {
        double x0 = a[i + 1], x1 = a[i + 2], x2 = a[i + 3], x3 = a[i + 4];
        double x4 = a[i + 5], x5 = a[i + 6], x6 = a[i + 7], x7 = a[i + 8];
        a[i]     = x0 + b[i];
        a[i + 1] = x1 + b[i + 1];
        a[i + 2] = x2 + b[i + 2];
        a[i + 3] = x3 + b[i + 3];
        a[i + 4] = x4 + b[i + 4];
        a[i + 5] = x5 + b[i + 5];
        a[i + 6] = x6 + b[i + 6];
        a[i + 7] = x7 + b[i + 7];
    }
    for (; i < hi; i++) {
        a[i] = a[i + 1] + b[i];
    }
    a[hi] = boundary + b[hi];
}

void ext_war_unit_fp64(double *a, double *b, int64_t LEN_1D,
                       uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D <= 1) return;
    int64_t n = LEN_1D - 1;

#ifdef _OPENMP
    if (n >= PAR_THRESHOLD) {
        int64_t nblocks = (n + BLOCK - 1) / BLOCK;
        double *boundary = (double *)malloc((size_t)nblocks * sizeof(double));
        if (boundary != NULL) {
            int64_t *lo_k = NULL;  /* not needed; recompute below */
            (void)lo_k;
            #pragma omp parallel
            {
                #pragma omp for schedule(static)
                for (int64_t k = 0; k < nblocks; k++) {
                    int64_t lo = k * (int64_t)BLOCK;
                    int64_t hi = lo + (int64_t)BLOCK - 1;
                    if (hi >= n) hi = n - 1;
                    boundary[k] = a[hi + 1];
                }
                #pragma omp for schedule(static)
                for (int64_t k = 0; k < nblocks; k++) {
                    int64_t lo = k * (int64_t)BLOCK;
                    int64_t hi = lo + (int64_t)BLOCK - 1;
                    if (hi >= n) hi = n - 1;
                    block_work(a, b, lo, hi, boundary[k]);
                }
            }
            free(boundary);
            return;
        }
    }
#endif
    /* serial fallback (also the whole path for small n):
       a[n] is never written, so it can serve as the boundary directly. */
    block_work(a, b, 0, n - 1, a[n]);
}
