#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static inline double bits_to_d(long long v) { union { long long l; double d; } u; u.l = v; return u.d; }
static inline long long d_to_bits(double d) { union { long long l; double d; } u; u.d = d; return u.l; }
static inline void dadd(double *p, double s) {
    long long old = *(volatile long long *)p;
    for (;;) {
        long long nv = d_to_bits(bits_to_d(old) + s);
        if (__atomic_compare_exchange_n((long long *)p, &old, nv, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
            return;
    }
}

void scatter_accum_dup_fp64(double *restrict bins, const int32_t *restrict ip,
                            const double *restrict src, const int64_t LEN_1D,
                            void *ws, int64_t ws_bytes) {
    int64_t n = LEN_1D;
    double *bc = (double *)ws;
    double t;

    memcpy(bc, bins, (size_t)n * 8);
    t = omp_get_wtime();
#pragma omp parallel for schedule(static, 512)
    for (int64_t i = 0; i + 8 <= n; i += 8) {
        int32_t v0 = ip[i], v1 = ip[i+1], v2 = ip[i+2], v3 = ip[i+3];
        int32_t v4 = ip[i+4], v5 = ip[i+5], v6 = ip[i+6], v7 = ip[i+7];
        dadd(&bins[v0], src[i]); dadd(&bins[v1], src[i+1]); dadd(&bins[v2], src[i+2]); dadd(&bins[v3], src[i+3]);
        dadd(&bins[v4], src[i+4]); dadd(&bins[v5], src[i+5]); dadd(&bins[v6], src[i+6]); dadd(&bins[v7], src[i+7]);
    }
    printf("dadd8_on_bins  = %.1fms\n", (omp_get_wtime() - t) * 1e3);

    t = omp_get_wtime();
#pragma omp parallel for schedule(static, 512)
    for (int64_t i = 0; i + 8 <= n; i += 8) {
        int32_t v0 = ip[i], v1 = ip[i+1], v2 = ip[i+2], v3 = ip[i+3];
        int32_t v4 = ip[i+4], v5 = ip[i+5], v6 = ip[i+6], v7 = ip[i+7];
        dadd(&bc[v0], src[i]); dadd(&bc[v1], src[i+1]); dadd(&bc[v2], src[i+2]); dadd(&bc[v3], src[i+3]);
        dadd(&bc[v4], src[i+4]); dadd(&bc[v5], src[i+5]); dadd(&bc[v6], src[i+6]); dadd(&bc[v7], src[i+7]);
    }
    printf("dadd8_on_ws    = %.1fms\n", (omp_get_wtime() - t) * 1e3);

    memcpy(bc, bins, (size_t)n * 8);
    t = omp_get_wtime();
#pragma omp parallel for schedule(static, 256)
    for (int64_t i = 0; i < n; ++i) {
#pragma omp atomic
        bins[ip[i]] += src[i];
    }
    printf("ompatomic_bins = %.1fms\n", (omp_get_wtime() - t) * 1e3);

    t = omp_get_wtime();
#pragma omp parallel for schedule(static, 256)
    for (int64_t i = 0; i < n; ++i) {
#pragma omp atomic
        bc[ip[i]] += src[i];
    }
    printf("ompatomic_ws   = %.1fms\n", (omp_get_wtime() - t) * 1e3);
    fflush(stdout);
}
