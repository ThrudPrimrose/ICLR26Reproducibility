#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

/* Blocked affine-map scan for y[i] = c[i]*y[i-1] + x[i], y[0] = x[0].
 *
 * A = (c[i]*y[i-1], +x[i]) is an affine map; maps compose associatively,
 * so the array is a prefix scan over affine maps. Plan:
 *   tiles of T*K*S elements; per tile:
 *     P1: each thread scans K of its blocks (independent (A,B) chain pairs)
 *         and records the block map (A_b, B_b).
 *     Q : serial scan of the K*T block pairs into carry maps (tiny).
 *     P3: each thread scans its K blocks (independent y chains) seeded by
 *         the carry; c/x were just read by P1 -> L2 hot; y via non-temporal
 *         stores (no read-for-ownership).
 */

#define KPAIR 4
#define PF 2048

static double *g_pa = NULL;
static double *g_pb = NULL;
static int64_t g_cap = 0;

static void get_bufs(int64_t m)
{
    if (m > g_cap) {
        int64_t nc = m < (1 << 20) ? (1 << 20) : m;
        double *na = (double *)realloc(g_pa, (size_t)nc * sizeof(double));
        double *nb = (double *)realloc(g_pb, (size_t)nc * sizeof(double));
        if (na && nb) {
            g_pa = na;
            g_pb = nb;
            g_cap = nc;
        }
    }
}

static inline void stnt16(double *p, double lo, double hi)
{
    _mm_stream_pd(p, _mm_set_pd(hi, lo));
}

void scan_affine_decay_fp64(double *c, double *x, double *y, const int64_t LEN_1D,
                            uint8_t *workspace, const int64_t workspace_bytes)
{
    int64_t n = LEN_1D;
    if (n <= 0)
        return;
    y[0] = x[0];
    if (n == 1)
        return;

    int nt = omp_get_max_threads();
    if (nt < 1)
        nt = 1;

    if ((int64_t)nt <= 1 || n < 32768) {
        double *restrict yy = y;
        const double *restrict cc = c;
        const double *restrict xx = x;
        for (int64_t i = 1; i < n; ++i)
            yy[i] = cc[i] * yy[i - 1] + xx[i];
        return;
    }

    int64_t S = 8192;
    if (n < (int64_t)nt * KPAIR * S * 4)
        S = 2048;
    int64_t m = (n - 1 + S - 1) / S;

    double *pa = NULL, *pb = NULL;
    if (workspace && workspace_bytes >= 2 * m * (int64_t)sizeof(double) && (uintptr_t)workspace % 32 == 0) {
        pa = (double *)workspace;
        pb = pa + m;
    } else {
        get_bufs(m);
        pa = g_pa;
        pb = g_pb;
    }

    double y0 = y[0];

#pragma omp parallel shared(c, x, y, pa, pb, m, n, S, y0, nt)
    {
        int tid = omp_get_thread_num();
        /* tile boundaries in block numbers */
        int64_t tile_blocks = (int64_t)nt * KPAIR;
        int64_t QA = 1.0, QB = 0.0;
        for (int64_t b0 = 0; b0 < m; b0 += tile_blocks) {
            int64_t b1 = b0 + tile_blocks;
            if (b1 > m)
                b1 = m;

            /* ---- P1: block maps ---- */
            for (int64_t b = b0 + tid; b < b1; b += nt) {
                int64_t base = 1 + b * S;
                int64_t end = base + S;
                if (end > n)
                    end = n;
                double A = 1.0, B = 0.0;
                int64_t i = base;
                for (; i < end; ++i) {
                    if (i + PF < end) {
                        _mm_prefetch((const char *)(c + i + PF), _MM_HINT_T0);
                        _mm_prefetch((const char *)(x + i + PF), _MM_HINT_T0);
                    }
                    double ci = c[i];
                    B = ci * B + x[i];
                    A = ci * A;
                }
                pa[b] = A;
                pb[b] = B;
            }
#pragma omp barrier
            /* ---- Q: serial carry scan of this tile ---- */
            if (tid == 0) {
                for (int64_t b = b0; b < b1; ++b) {
                    double Pa = pa[b], Pb = pb[b];
                    pa[b] = QA;
                    pb[b] = QB;
                    QA = Pa * QA;
                    QB = Pa * QB + Pb;
                }
            }
#pragma omp barrier
            /* ---- P3: seeded scan ---- */
            for (int64_t b = b0 + tid; b < b1; b += nt) {
                int64_t base = 1 + b * S;
                int64_t end = base + S;
                if (end > n)
                    end = n;
                double prev = y0 * pa[b] + pb[b];
                int64_t i = base;
                /* peel to 16B-aligned y store address */
                for (; i < end && (((uintptr_t)(void *)(uintptr_t)(y + i)) & 15U) != 0; ++i) {
                    prev = c[i] * prev + x[i];
                    y[i] = prev;
                }
                for (; i + 1 < end; i += 2) {
                    if (i + PF < end) {
                        _mm_prefetch((const char *)(c + i + PF), _MM_HINT_T0);
                        _mm_prefetch((const char *)(x + i + PF), _MM_HINT_T0);
                    }
                    double p0 = c[i] * prev + x[i];
                    double p1 = c[i + 1] * p0 + x[i + 1];
                    stnt16(y + i, p0, p1);
                    prev = p1;
                }
                for (; i < end; ++i) {
                    prev = c[i] * prev + x[i];
                    y[i] = prev;
                }
            }
        }
    }
}
