#include <stddef.h>
#include <stdint.h>
#include <omp.h>

#define MAX_THREADS 256

void scan_affine_decay_fp64(double *restrict y,
                              double *restrict c,
                              double *restrict x,
                              int64_t LEN_1D,
                              uint8_t *restrict workspace,
                              int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;

    if (LEN_1D <= 0)
        return;

    y[0] = x[0];
    if (LEN_1D == 1)
        return;

    /* Serial fallback for small inputs. */
    if (LEN_1D < 4096) {
        for (int64_t i = 1; i < LEN_1D; ++i)
            y[i] = c[i] * y[i - 1] + x[i];
        return;
    }

    int nthreads = omp_get_max_threads();
    if (nthreads > MAX_THREADS)
        nthreads = MAX_THREADS;
    if (nthreads > (int)LEN_1D)
        nthreads = (int)LEN_1D;

    int64_t block = (LEN_1D + nthreads - 1) / nthreads;

    /* Per-thread affine carries. */
    double P[MAX_THREADS];
    double S[MAX_THREADS];

    /* Phase 1: each block computes a local scan (prefix = 0) and its c-product. */
    #pragma omp parallel for schedule(static)
    for (int tid = 0; tid < nthreads; ++tid) {
        int64_t lo = (int64_t)tid * block;
        int64_t hi = lo + block;
        if (hi > LEN_1D)
            hi = LEN_1D;

        double local_y = x[lo];
        y[lo] = local_y;
        double prod = c[lo];

        for (int64_t i = lo + 1; i < hi; ++i) {
            local_y = c[i] * local_y + x[i];
            y[i] = local_y;
            prod *= c[i];
        }

        P[tid] = prod;
        S[tid] = local_y;
    }

    /* Phase 2: serial combine block prefixes. S[b] becomes y[end of block b]. */
    for (int b = 1; b < nthreads; ++b) {
        S[b] = P[b] * S[b - 1] + S[b];
        P[b] = P[b] * P[b - 1];
    }

    /* Phase 3: apply cumulative block prefix to every element. */
    #pragma omp parallel for schedule(static)
    for (int tid = 0; tid < nthreads; ++tid) {
        double prefix = (tid == 0) ? 0.0 : S[tid - 1];

        int64_t lo = (int64_t)tid * block;
        int64_t hi = lo + block;
        if (hi > LEN_1D)
            hi = LEN_1D;

        double running = 1.0;
        for (int64_t i = lo; i < hi; ++i) {
            running *= c[i];
            y[i] = running * prefix + y[i];
        }
    }
}
