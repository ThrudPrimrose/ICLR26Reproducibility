#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void versioned_distance_update_fp64(double *restrict a,
                                    const double *restrict b,
                                    const double *restrict c,
                                    const int64_t K,
                                    const int64_t LEN_1D) {
    if (K <= 0 || K >= LEN_1D) {
        return;
    }

    if (K == 1) {
        int64_t N = LEN_1D;
        int nthreads = omp_get_max_threads();
        if (N <= 4096 || nthreads <= 1) {
            double x = a[0];
            for (int64_t i = 1; i < N; ++i) {
                x = 0.75 * x + b[i] * c[i];
                a[i] = x;
            }
            return;
        }

        const double p = 0.75;
        const int64_t MIN_CHUNK = 512;
        int64_t C = nthreads;
        int64_t L = (N + C - 1) / C;
        if (L < MIN_CHUNK) {
            L = MIN_CHUNK;
            C = (N + L - 1) / L;
        }

        double add[C];

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int64_t m = 0; m < C; ++m) {
                int64_t s = m * L;
                int64_t e = s + L;
                if (e > N) e = N;
                double x = 0.0;
                for (int64_t i = s + 1; i < e; ++i) {
                    x = p * x + b[i] * c[i];
                }
                add[m] = x;
            }

            #pragma omp single
            {
                for (int64_t m = C - 1; m >= 1; --m) {
                    int64_t idx = m * L;
                    add[m] = p * add[m - 1] + b[idx] * c[idx];
                }
                add[0] = a[0];
            }

            #pragma omp for schedule(static)
            for (int64_t m = 0; m < C; ++m) {
                int64_t s = m * L;
                int64_t e = s + L;
                if (e > N) e = N;
                double x = add[m];
                a[s] = x;
                for (int64_t i = s + 1; i < e; ++i) {
                    x = p * x + b[i] * c[i];
                    a[i] = x;
                }
            }
        }
        return;
    }

    /* K > 1: block-parallel scan.  Each block of K elements depends only on
       the previous block, so we treat blocks as vector states and parallelise
       across chunks of blocks.  The 0.75 decay makes the chunk starts exact in
       double arithmetic for chunk sizes above a K-dependent minimum. */
    int64_t N = LEN_1D;
    int64_t B_full = N / K;          /* number of full K-blocks (incl. seed block 0) */
    int64_t R = N - B_full * K;      /* remainder elements in final partial block */

    /* If there is nothing beyond the seed block, just handle the tail. */
    if (B_full <= 1) {
        if (R > 0) {
            int64_t base = B_full * K;
            #pragma omp simd
            for (int64_t i = 0; i < R; ++i) {
                a[base + i] = 0.75 * a[base - K + i] + b[base + i] * c[base + i];
            }
        }
        return;
    }

    int nthreads = omp_get_max_threads();
    /* The recurrence between consecutive K-blocks multiplies by 0.75, so we
       need the same ~512 block history that the K==1 path uses to make the
       zero-start leave block equal to the real one in double arithmetic. */
    const int64_t min_blocks = 512;

    int64_t M_full = (B_full + nthreads - 1) / nthreads;
    if (M_full < min_blocks) M_full = min_blocks;
    int64_t C = (B_full + M_full - 1) / M_full;

    /* Fall back to serial when the array is too small to benefit. */
    if (C <= 1 || nthreads <= 1) {
        for (int64_t base = K; base < N; base += K) {
            int64_t end = base + K;
            if (end > N) end = N;
            #pragma omp simd
            for (int64_t i = base; i < end; ++i) {
                a[i] = 0.75 * a[i - K] + b[i] * c[i];
            }
        }
        return;
    }

    size_t wsz = (size_t)C * (size_t)K * sizeof(double);
    wsz = (wsz + 63) & ~((size_t)63);
    double *restrict leave = aligned_alloc(64, wsz);
    double *restrict start = aligned_alloc(64, wsz);
    const double p = 0.75;

    #pragma omp parallel
    {
        /* Pass 1: leave vector of each chunk, assuming the chunk's first
           block is zero.  Because K*(M_full-1) is large enough, this equals
           the real leave vector in double arithmetic. */
        #pragma omp for schedule(static)
        for (int64_t m = 0; m < C; ++m) {
            int64_t block_s = m * M_full;
            int64_t block_e = block_s + M_full;
            if (block_e > B_full) block_e = B_full;
            double x[K];
            for (int64_t i = 0; i < K; ++i) x[i] = 0.0;
            for (int64_t t = block_s + 1; t < block_e; ++t) {
                int64_t base = t * K;
                #pragma omp simd
                for (int64_t i = 0; i < K; ++i) {
                    x[i] = p * x[i] + b[base + i] * c[base + i];
                }
            }
            for (int64_t i = 0; i < K; ++i) {
                leave[m * K + i] = x[i];
            }
        }

        /* Pass 2: serial prefix scan over chunk boundary blocks. */
        #pragma omp single
        {
            for (int64_t i = 0; i < K; ++i) {
                start[i] = a[i];
            }
            for (int64_t m = 1; m < C; ++m) {
                int64_t base = m * M_full * K;
                for (int64_t i = 0; i < K; ++i) {
                    start[m * K + i] = p * leave[(m - 1) * K + i]
                                         + b[base + i] * c[base + i];
                }
            }
        }

        /* Pass 3: write all full blocks from exact chunk starts. */
        #pragma omp for schedule(static)
        for (int64_t m = 0; m < C; ++m) {
            int64_t block_s = m * M_full;
            int64_t block_e = block_s + M_full;
            if (block_e > B_full) block_e = B_full;
            int64_t elem_s = block_s * K;
            for (int64_t i = 0; i < K; ++i) {
                a[elem_s + i] = start[m * K + i];
            }
            for (int64_t t = block_s + 1; t < block_e; ++t) {
                int64_t base = t * K;
                #pragma omp simd
                for (int64_t i = 0; i < K; ++i) {
                    a[base + i] = p * a[base - K + i]
                                  + b[base + i] * c[base + i];
                }
            }
        }
    }

    /* Final partial block, if any. */
    if (R > 0) {
        int64_t base = B_full * K;
        #pragma omp simd
        for (int64_t i = 0; i < R; ++i) {
            a[base + i] = p * a[base - K + i] + b[base + i] * c[base + i];
        }
    }

    free(leave);
    free(start);
}
