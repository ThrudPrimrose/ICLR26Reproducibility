/* TSVC s231:  aa[j,i] = aa[j-1,i] + bb[j,i]   (0 <= i < N, 1 <= j < N)
 *
 * The dependence runs down columns (j direction); the i dimension is fully
 * independent.  Scheme: partition the column range statically across OpenMP
 * threads (one contiguous band per thread); each thread walks all rows of
 * its band.  Per row the access is unit-stride on both arrays, so the inner
 * loop is a straight vector stream; the aa[j-1] read is an L2 hit on the row
 * the thread itself just wrote.  No barriers beyond team enter/exit.
 *
 * ABI (judge binding): pointers sorted by name, then size symbols, then the
 * trailing workspace pair (unused): (aa, bb, LEN_2D).
 */
#include <stdint.h>
#include <omp.h>

/* 0 = use OMP_NUM_THREADS; else force this team size. */
#ifndef S231_NTHREADS
#define S231_NTHREADS 24
#endif

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb,
                      int64_t LEN_2D)
{
    const int64_t N = LEN_2D;
    if (N <= 1)
        return;

#if S231_NTHREADS > 0
    #pragma omp parallel num_threads(S231_NTHREADS)
#else
    #pragma omp parallel
#endif
    {
        const int nt = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t c0 = (N * tid) / nt;
        const int64_t c1 = (N * (tid + 1)) / nt;
        if (c0 < c1) {
            const int64_t len = c1 - c0;
            for (int64_t j = 1; j < N; ++j) {
                double *a1 = aa + j * N + c0;
                const double *a0 = a1 - N;
                const double *b = bb + j * N + c0;
                if (j + 2 < N) {
                    const double *b1 = b + N;
                    const double *b2 = b + 2 * N;
                    for (int64_t q = 0; q < len; q += 16) {
                        __builtin_prefetch(b1 + q, 0, 3);
                        __builtin_prefetch(b2 + q, 0, 3);
                    }
                }
                int64_t k = 0;
                for (; k + 8 <= len; k += 8) {
                    a1[k + 0] = a0[k + 0] + b[k + 0];
                    a1[k + 1] = a0[k + 1] + b[k + 1];
                    a1[k + 2] = a0[k + 2] + b[k + 2];
                    a1[k + 3] = a0[k + 3] + b[k + 3];
                    a1[k + 4] = a0[k + 4] + b[k + 4];
                    a1[k + 5] = a0[k + 5] + b[k + 5];
                    a1[k + 6] = a0[k + 6] + b[k + 6];
                    a1[k + 7] = a0[k + 7] + b[k + 7];
                }
                for (; k < len; ++k)
                    a1[k] = a0[k] + b[k];
            }
        }
    }
}
