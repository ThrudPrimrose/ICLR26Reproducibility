#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static double *g_flux = NULL;
static size_t g_flux_cap = 0;

static inline double *flux_buf(size_t need) {
    if (need > g_flux_cap) {
        if (g_flux) free(g_flux);
        g_flux_cap = need;
        g_flux = (double *)aligned_alloc(64, (g_flux_cap + 63) & ~(size_t)63);
        if (!g_flux) g_flux = (double *)malloc(g_flux_cap);
    }
    return g_flux;
}

/* Private staging buffer for the small-N path: T slices of N doubles.
 * Persistent (grow-only) so repeated calls skip the alloc; re-zeroed in
 * parallel every call. */
static double *g_stage = NULL;
static size_t g_stage_elems = 0;  /* T*N for which the buffer was sized */

static inline void *staging_buf(size_t elems) {
    if (elems > g_stage_elems) {
        size_t need = elems * sizeof(double);
        if (g_stage) free(g_stage);
        g_stage_elems = elems;
        g_stage = (double *)aligned_alloc(64, (need + 63) & ~(size_t)63);
        if (!g_stage) g_stage = (double *)malloc(need);
    }
    return g_stage;
}

/* Zero `nbytes` starting at `p` (64B-aligned) across the caller's team. */
static inline void zero_parallel(void *p, size_t nbytes, int T) {
    size_t per = (nbytes / (size_t)T) & ~(size_t)7;
    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        size_t off = (size_t)t * per;
        size_t end = off + per;
        if (t == T - 1) end = nbytes;  /* T*per <= nbytes: last thread takes the whole tail */
        memset((char *)p + off, 0, end - off);
    }
}

static inline void flux_unroll(int64_t e, int64_t e1,
                               const int64_t *src, const int64_t *dst,
                               const double *w, const double *x, double *flux) {
    int64_t i;
    for (i = e; i + 8 <= e1; i += 8) {
        int64_t s0=src[i],   s1=src[i+1], s2=src[i+2], s3=src[i+3];
        int64_t s4=src[i+4], s5=src[i+5], s6=src[i+6], s7=src[i+7];
        int64_t d0=dst[i],   d1=dst[i+1], d2=dst[i+2], d3=dst[i+3];
        int64_t d4=dst[i+4], d5=dst[i+5], d6=dst[i+6], d7=dst[i+7];
        double f0 = w[i]   * (x[s0]-x[d0]);
        double f1 = w[i+1] * (x[s1]-x[d1]);
        double f2 = w[i+2] * (x[s2]-x[d2]);
        double f3 = w[i+3] * (x[s3]-x[d3]);
        double f4 = w[i+4] * (x[s4]-x[d4]);
        double f5 = w[i+5] * (x[s5]-x[d5]);
        double f6 = w[i+6] * (x[s6]-x[d6]);
        double f7 = w[i+7] * (x[s7]-x[d7]);
        flux[i]=f0; flux[i+1]=f1; flux[i+2]=f2; flux[i+3]=f3;
        flux[i+4]=f4; flux[i+5]=f5; flux[i+6]=f6; flux[i+7]=f7;
    }
    for (; i < e1; ++i) flux[i] = w[i] * (x[src[i]] - x[dst[i]]);
}

void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict dst,
                         const int64_t *restrict src, const double *restrict w,
                         const double *restrict x, const int64_t E, const int64_t N) {
    int T = omp_get_max_threads();
    if (T < 1) T = 1;

    if (N <= 200000) {
        size_t elems = (size_t)N * (size_t)T;
        double *stage = (double *)staging_buf(elems);
        zero_parallel(stage, elems * sizeof(double), T);

        #pragma omp parallel for schedule(static) num_threads(T)
        for (int64_t e = 0; e < E; e += 8) {
            int64_t e1 = e + 8 < E ? e + 8 : E;
            int64_t i;
            double *st = stage + (size_t)(omp_get_thread_num() * N);
            for (i = e; i + 8 <= e1; i += 8) {
                int64_t s0=src[i],s1=src[i+1],s2=src[i+2],s3=src[i+3],s4=src[i+4],s5=src[i+5],s6=src[i+6],s7=src[i+7];
                int64_t d0=dst[i],d1=dst[i+1],d2=dst[i+2],d3=dst[i+3],d4=dst[i+4],d5=dst[i+5],d6=dst[i+6],d7=dst[i+7];
                double f0=w[i]*(x[s0]-x[d0]);   double f1=w[i+1]*(x[s1]-x[d1]);
                double f2=w[i+2]*(x[s2]-x[d2]); double f3=w[i+3]*(x[s3]-x[d3]);
                double f4=w[i+4]*(x[s4]-x[d4]); double f5=w[i+5]*(x[s5]-x[d5]);
                double f6=w[i+6]*(x[s6]-x[d6]); double f7=w[i+7]*(x[s7]-x[d7]);
                st[s0]+=f0; st[d0]-=f0; st[s1]+=f1; st[d1]-=f1; st[s2]+=f2; st[d2]-=f2; st[s3]+=f3; st[d3]-=f3;
                st[s4]+=f4; st[d4]-=f4; st[s5]+=f5; st[d5]-=f5; st[s6]+=f6; st[d6]-=f6; st[s7]+=f7; st[d7]-=f7;
            }
            for (; i < e1; ++i) {
                int64_t s=src[i], d=dst[i];
                double f=w[i]*(x[s]-x[d]);
                st[s]+=f; st[d]-=f;
            }
        }

        #pragma omp parallel for schedule(static) num_threads(T)
        for (int64_t i = 0; i < N; ++i) {
            double s = 0.0;
            for (int t = 0; t < T; ++t) s += stage[(size_t)t * N + i];
            Lx[i] = s;
        }
        return;
    }

    double *flux = flux_buf((size_t)E * sizeof(double));
    zero_parallel(Lx, (size_t)N * sizeof(double), T);

    #pragma omp parallel for schedule(static) num_threads(T)
    for (int64_t e = 0; e < E; e += 8) {
        int64_t e1 = e + 8 < E ? e + 8 : E;
        flux_unroll(e, e1, src, dst, w, x, flux);
    }

    #pragma omp parallel for schedule(static) num_threads(T)
    for (int64_t e = 0; e < E; ++e) {
        #pragma omp atomic
        Lx[src[e]] += flux[e];
        #pragma omp atomic
        Lx[dst[e]] -= flux[e];
    }
}
