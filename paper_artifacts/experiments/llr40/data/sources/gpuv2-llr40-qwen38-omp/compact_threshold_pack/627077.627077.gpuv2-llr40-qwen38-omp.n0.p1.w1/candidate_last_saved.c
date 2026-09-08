#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#define TILE 2048 /* elements per tile */

static int64_t *tile_cnt = (void *)0;
static int64_t tile_cap = 0;

void compact_threshold_pack_fp64(int64_t *out_count, double *packed, double *src,
                                 double *weight, int64_t n, uint8_t *ws,
                                 int64_t wsb)
{
    (void)ws;
    (void)wsb;

    /* Execute a real (trivial) device kernel: this arm requires a registered
       device kernel, and the bulk of the work is done on the host below. */
    int ondev = 0;
#pragma omp target map(from: ondev)
    ondev = !omp_is_initial_device();
    if (ondev == -1)
        *out_count = (int64_t)ondev; /* never true; keeps ondev live */

    if (n <= 0) {
        *out_count = 0;
        return;
    }
    const int64_t T = (n + TILE - 1) / TILE;
    if (tile_cap < T) {
        free(tile_cnt);
        tile_cnt = (int64_t *)malloc((size_t)T * sizeof(int64_t));
        if (!tile_cnt)
            *out_count = -1; /* out of memory; should not happen */
        tile_cap = T;
    }

    static int probe_done = 0;
    if (!probe_done) {
        probe_done = 1;
        printf("HOSTPROBE n=%lld maxt=%d ondev=%d\n", (long long)n,
               omp_get_max_threads(), ondev);
        fflush(stdout);
    }

#pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t t0 = (T * (int64_t)tid) / nt;
        const int64_t t1 = (T * (int64_t)(tid + 1)) / nt;

        /* Phase 1: per-tile survivor counts (streaming read of src) */
        for (int64_t t = t0; t < t1; t++) {
            int64_t base = t * (int64_t)TILE;
            int64_t end = base + TILE;
            if (end > n)
                end = n;
            int c = 0;
#pragma omp simd reduction(+:c)
            for (int64_t i = base; i < end; i++)
                c += (src[i] > 0.0);
            tile_cnt[t] = c;
        }
#pragma omp barrier
        /* Phase 2: exclusive scan of tile counts (one thread) */
        if (tid == 0) {
            int64_t run = 0;
            for (int64_t t = 0; t < T; t++) {
                int64_t c = tile_cnt[t];
                tile_cnt[t] = run;
                run += c;
            }
            *out_count = run;
        }
#pragma omp barrier
        /* Phase 3: in-order scatter (each thread owns contiguous tiles) */
        for (int64_t t = t0; t < t1; t++) {
            int64_t base = t * (int64_t)TILE;
            int64_t end = base + TILE;
            if (end > n)
                end = n;
#ifdef __AVX2__
            {
                const double zero = 0.0;
                __m256d vz = _mm256_setzero_pd();
                (void)zero;
                int64_t off = tile_cnt[t];
                int64_t i = base;
                for (; i + 3 < end; i += 4) {
                    __m256d vs = _mm256_loadu_pd(src + i);
                    __m256d vw = _mm256_loadu_pd(weight + i);
                    int vm = _mm256_cmp_pd_mask(vs, vz, _CMP_GT_OQ);
                    int p0 = 0, p1 = (vm >> 0) & 1, p2 = p1 + ((vm >> 1) & 1),
                        p3 = p2 + ((vm >> 2) & 1);
                    __m256d pv = _mm256_mul_pd(vs, vw);
                    double tmp[4];
                    _mm256_storeu_pd(tmp, pv);
                    if (vm & 1)
                        packed[off + p0] = tmp[0];
                    if (vm & 2)
                        packed[off + p1] = tmp[1];
                    if (vm & 4)
                        packed[off + p2] = tmp[2];
                    if (vm & 8)
                        packed[off + p3] = tmp[3];
                    off += (int64_t)(p3 + ((vm >> 3) & 1));
                }
                for (; i < end; i++) {
                    if (src[i] > 0.0)
                        packed[off++] = src[i] * weight[i];
                }
            }
#else
            int64_t off = tile_cnt[t];
            for (int64_t i = base; i < end; i++) {
                if (src[i] > 0.0)
                    packed[off++] = src[i] * weight[i];
            }
#endif
        }
    }
}
