/* Wavefront difference-diagonal skew, fp64, in-place.
 *
 * a[i][j] += a[i-1][j] + a[i-1][j+1],  i = 1..N-1, j = 0..N-2.
 *
 * Structure: rows form a serial chain (the critical path); within one row
 * the columns are independent.  The matrix is cut into contiguous column
 * slabs, one per thread (64 B aligned borders).  The only cross-slab
 * dependency per row is ONE column: slab t needs a[i-1][c0+W] from slab
 * t+1, so slab t waits for its right neighbour to finish the row, does
 * the row, and publishes a per-row counter (release/acquire, one 64 B
 * cache line per thread).
 *
 * Batching: B rows per wait/hop.  Slab t computes rows [i0, i1) after the
 * right neighbour has published row i1-1 (which also covers rows i0-1 ..
 * i1-2).  This amortises the 24-hop wave latency over B rows of work.
 *
 * The row pair window (2 rows x N x 8 B) is far smaller than the LLC of
 * the 24 cores, so steady state is L2/L3 resident; a prefetch runs ~16
 * rows ahead.
 *
 * FP: per-element order (cur + prev) + shift, left-associative --
 * bit-identical to the scalar reference.
 */
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>
#include <omp.h>

#define WF_ROW_BITS 20                       /* rows per epoch must stay below 2^20 */
#define WF_MAX_THREADS 256
#define WF_BATCH 4                           /* rows per hop */
#define WF_PREFETCH_ROWS 16

static _Alignas(64) uint64_t g_cnt[WF_MAX_THREADS][8];
static _Atomic uint64_t g_epoch = 0;

/* cur[j..j+31] += prev[j..j+31] + prev[j+1..j+32]  (order preserved) */
static inline void wf_row32(double *restrict cur, const double *restrict prev,
                            int64_t m)
{
    int64_t j = 0;
    for (; j + 32 <= m; j += 32) {
        __m512d c0 = _mm512_loadu_pd(cur + j);
        __m512d c1 = _mm512_loadu_pd(cur + j + 8);
        __m512d c2 = _mm512_loadu_pd(cur + j + 16);
        __m512d c3 = _mm512_loadu_pd(cur + j + 24);
        __m512d p0 = _mm512_loadu_pd(prev + j);
        __m512d p1 = _mm512_loadu_pd(prev + j + 8);
        __m512d p2 = _mm512_loadu_pd(prev + j + 16);
        __m512d p3 = _mm512_loadu_pd(prev + j + 24);
        __m512d s0 = _mm512_loadu_pd(prev + j + 1);
        __m512d s1 = _mm512_loadu_pd(prev + j + 9);
        __m512d s2 = _mm512_loadu_pd(prev + j + 17);
        __m512d s3 = _mm512_loadu_pd(prev + j + 25);
        c0 = _mm512_add_pd(c0, p0);
        c0 = _mm512_add_pd(c0, s0);
        c1 = _mm512_add_pd(c1, p1);
        c1 = _mm512_add_pd(c1, s1);
        c2 = _mm512_add_pd(c2, p2);
        c2 = _mm512_add_pd(c2, s2);
        c3 = _mm512_add_pd(c3, p3);
        c3 = _mm512_add_pd(c3, s3);
        _mm512_storeu_pd(cur + j, c0);
        _mm512_storeu_pd(cur + j + 8, c1);
        _mm512_storeu_pd(cur + j + 16, c2);
        _mm512_storeu_pd(cur + j + 24, c3);
    }
    for (; j < m; j++)
        cur[j] = cur[j] + prev[j] + prev[j + 1];
}

static inline void wf_prefetch_row(const double *row, int64_t m)
{
    int64_t j = 0;
    for (; j + 8 <= m; j += 8)
        _mm_prefetch((const char *)(row + j), _MM_HINT_T0);
    for (; j < m; j++)
        _mm_prefetch((const char *)(row + j), _MM_HINT_T0);
}

void wf_diff_skew_fp64(double *restrict a, int64_t LEN_2D,
                       uint8_t *restrict workspace, int64_t workspace_size)
{
    const int64_t N = LEN_2D;
    (void)workspace;
    (void)workspace_size;
    if (N <= 1) return;

#pragma omp parallel
    {
        const int t = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int act = (nt < WF_MAX_THREADS) ? nt : WF_MAX_THREADS;
        if (t < act) {
            const uint64_t epoch = g_epoch;
            const int64_t W = (((N - 1) + act - 1) / act + 7) & ~7;
            const int64_t c0 = (int64_t)t * W;
            int64_t m = W;
            if (m > N - 1 - c0) m = N - 1 - c0;
            if (m < 0) m = 0;
            /* number of non-empty slabs: slabs past it stay out of the
             * chain so their hops never tax the critical path */
            const int R = (int)(((N - 1) + W - 1) / W);
            if (t < R) {
                const uint64_t hi = epoch << WF_ROW_BITS;
                uint64_t *const my_cnt = &g_cnt[t][0];
                const int right = (t + 1 < R) ? t + 1 : -1;

                for (int64_t r = 1; r <= WF_PREFETCH_ROWS && r < N; r++)
                    wf_prefetch_row(a + r * N + c0, m);

                for (int64_t i0 = 1; i0 < N; i0 += WF_BATCH) {
                    const int64_t i1 = (i0 + WF_BATCH < N) ? i0 + WF_BATCH : N;
                    if (right >= 0) {
                        const uint64_t need = hi | (uint64_t)(i1 - 1);
                        uint64_t v;
                        unsigned burn = 0x9e3779b9u;
                        while ((v = __atomic_load_n(&g_cnt[right][0],
                                                    __ATOMIC_ACQUIRE)) < need) {
                            /* keep the core hot between polls so C-state
                             * exit latency never lands on the publish */
                            for (int k = 0; k < 32; k++)
                                __asm__ volatile ("mov %0, %0" : "+r"(burn));
                            __builtin_ia32_pause();
                        }
                        (void)v;
                    }
                    for (int64_t i = i0; i < i1; i++) {
                        double *restrict cur = a + i * N + c0;
                        const double *restrict prev = a + (i - 1) * N + c0;
                        wf_row32(cur, prev, m);
                    }
                    __atomic_store_n(my_cnt, hi | (uint64_t)(i1 - 1),
                                     __ATOMIC_RELEASE);
                    const int64_t pf = i1 + WF_PREFETCH_ROWS;
                    if (pf < N)
                        wf_prefetch_row(a + pf * N + c0, m);
                }
            }
        }
    }
    g_epoch++;
}
