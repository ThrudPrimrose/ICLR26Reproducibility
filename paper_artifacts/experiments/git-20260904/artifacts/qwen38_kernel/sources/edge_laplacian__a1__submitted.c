#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#define NTMAX 32
#define UNR 16

static double *parts[NTMAX] = {0};
static int64_t parts_N = 0;
static int parts_nt = 0;

void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict src,
                         const int64_t *restrict dst, const double *restrict w,
                         const double *restrict x, int64_t E, int64_t N)
{
    if (E <= 0 || N <= 0) return;
    int nt = omp_get_max_threads();
    if (nt > NTMAX) nt = NTMAX;
    if (nt < 1) nt = 1;
    if (parts_N < N || parts_nt < nt) {
        int64_t nn = parts_N < N ? N : parts_N;
        for (int t = 0; t < NTMAX; t++) free(parts[t]);
        for (int t = 0; t < NTMAX; t++) parts[t] = (double*)calloc((size_t)nn, sizeof(double));
        parts_N = nn; parts_nt = nt;
    }
    #pragma omp parallel num_threads(nt)
    {
        double *L = parts[omp_get_thread_num()];
        #pragma omp for schedule(static)
        for (int64_t e0 = 0; e0 < E; e0 += UNR) {
            const int64_t m = (e0 + UNR <= E) ? UNR : E - e0;
            if (e0 + 32 < E) {
                __builtin_prefetch(src + e0 + 16, 0, 3);
                __builtin_prefetch(dst + e0 + 16, 0, 3);
                __builtin_prefetch(w   + e0 + 16, 0, 3);
            }

            int64_t s[UNR] = {0}; int64_t d[UNR] = {0}; double wv[UNR] = {0};
            #pragma GCC unroll 16
            for (int k = 0; k < UNR; k++)
                if (k < m) { s[k] = src[e0+k]; d[k] = dst[e0+k]; wv[k] = w[e0+k]; }
            #pragma GCC unroll 16
            for (int k = 0; k < UNR; k++)
                if (k < m) {
                    const double fx = wv[k] * (x[s[k]] - x[d[k]]);
                    L[s[k]] += fx;
                    L[d[k]] -= fx;
                }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < N; i0 += 8) {
        const int64_t m = (i0 + 8 <= N) ? 8 : N - i0;
        if (m == 8) {
            __m512d vsum = _mm512_loadu_pd(parts[0] + i0);
            for (int t = 1; t < nt; t++) {
                const __m512d vt = _mm512_loadu_pd(parts[t] + i0);
                vsum = _mm512_add_pd(vsum, vt);
                _mm512_storeu_pd(parts[t] + i0, _mm512_setzero_pd());
            }
            _mm512_storeu_pd(Lx + i0, vsum);
            _mm512_storeu_pd(parts[0] + i0, _mm512_setzero_pd());
        } else {
            for (int k = 0; k < m; k++) {
                double a = 0.0;
                for (int t = 0; t < nt; t++) { a += parts[t][i0+k]; parts[t][i0+k] = 0.0; }
                Lx[i0+k] = a;
            }
        }
    }
    for (int64_t i = N; i < parts_N; i++)
        for (int t = 0; t < nt; t++) parts[t][i] = 0.0;
}
