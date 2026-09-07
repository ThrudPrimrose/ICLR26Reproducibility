#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#ifndef V7_TMAX
#define V7_TMAX 24
#endif

static double **v3_bufs = NULL;
static int64_t v3_bufN = 0;
static int v3_bufT = 0;
#define SCRATCH_CAP (48LL * 1024 * 1024 * 1024)

static void ensure_v3(int64_t N, int T) {
    if (N <= v3_bufN && T <= v3_bufT) return;
    for (int i = 0; i < v3_bufT; ++i) free(v3_bufs[i]);
    free(v3_bufs);
    v3_bufs = NULL; v3_bufN = 0; v3_bufT = 0;
    int64_t bytes_per = N * 8;
    int64_t Tc = SCRATCH_CAP / (bytes_per > 0 ? bytes_per : 1);
    if (Tc < 1) Tc = 1;
    if (T > Tc) T = (int)Tc;
    v3_bufs = (double **)calloc((size_t)T, sizeof(double *));
    v3_bufT = T; v3_bufN = N;
    for (int i = 0; i < T; ++i) {
        v3_bufs[i] = (double *)malloc((size_t)N * sizeof(double));
        if (!v3_bufs[i]) { v3_bufT = i; v3_bufs = (double **)realloc(v3_bufs, (size_t)i * sizeof(double *)); return; }
    }
}

static void v3_path(double *restrict Lx, const int64_t *restrict dst,
                    const int64_t *restrict src, const double *restrict w,
                    const double *restrict x, int64_t E, int64_t N, int T) {
    if (T <= 1) {
        for (int64_t i = 0; i < N; ++i) Lx[i] = 0.0;
        for (int64_t i = 0; i < E; ++i) {
            double f = w[i] * (x[src[i]] - x[dst[i]]);
            Lx[src[i]] += f;
            Lx[dst[i]] -= f;
        }
        return;
    }
    ensure_v3(N, T);
    if (T > v3_bufT) T = v3_bufT;
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        int64_t lo = E * (int64_t)tid / (int64_t)T;
        int64_t hi = E * (int64_t)(tid + 1) / (int64_t)T;
        double *buf = v3_bufs[tid];
        for (int64_t i = 0; i < N; ++i) buf[i] = 0.0;
        for (int64_t i = lo; i < hi; ++i) {
            double f = w[i] * (x[src[i]] - x[dst[i]]);
            buf[src[i]] += f;
            buf[dst[i]] -= f;
        }
    }
    #pragma omp parallel num_threads(T)
    {
        #pragma omp for schedule(static, 32768)
        for (int64_t i = 0; i < N; ++i) Lx[i] = v3_bufs[0][i];
        for (int t = 1; t < T; ++t) {
            double *bt = v3_bufs[t];
            #pragma omp for schedule(static, 32768)
            for (int64_t i = 0; i < N; ++i) Lx[i] += bt[i];
        }
    }
}

/* ---------------- CSR state ---------------- */
static int32_t *cntOut = NULL, *cntIn = NULL;   /* [T][N] transient build slabs */
static int32_t *outIdx = NULL, *inIdx = NULL;   /* [E] transient */
static int32_t *offOut = NULL, *offIn = NULL;   /* [N+1] persistent */
static int32_t *posOfOut = NULL, *inPos = NULL; /* inPos persistent; posOfOut transient */
static int32_t *srcOut = NULL, *dstOut = NULL;  /* [E] persistent, out-ordered node ids */
static double *wOut = NULL;                     /* [E] persistent, out-ordered */
static double *F = NULL;                        /* [E] persistent, flux by out-position */
static int64_t csr_Ecap = 0, csr_Ncap = 0;      /* capacities of E/N-sized arrays */
static int64_t csr_T = 0, csr_N = 0, csr_E = 0;
static const int64_t *k_src = NULL, *k_dst = NULL;
static const double *k_w = NULL, *k_x = NULL;
static uint64_t k_hash = 0;
static int csr_valid = 0;

static inline uint64_t d64_bits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

static uint64_t sample_hash(const int64_t *a, const int64_t *b, const double *c, const double *d,
                            int64_t E, int64_t N) {
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ (uint64_t)E ^ ((uint64_t)N << 1);
    int64_t step = E > 4096 ? E / 4096 : 1;
    for (int64_t j = 0; j < E; j += step) {
        h ^= (uint64_t)a[j] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= (uint64_t)b[j] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= d64_bits(c[j]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    step = N > 2048 ? N / 2048 : 1;
    for (int64_t j = 0; j < N; j += step)
        h ^= d64_bits(d[j]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

static int csr_ensure(int64_t E, int64_t N, int T,
                      const int64_t *src, const int64_t *dst,
                      const double *w, const double *x) {
    uint64_t h = sample_hash(src, dst, w, x, E, N);
    if (csr_valid && csr_N == N && csr_E == E && csr_T == T &&
        k_src == src && k_dst == dst && k_w == w && k_x == x && k_hash == h)
        return 1;
    /* (re)build */
    if (csr_T != T || csr_N != N) {
        free(offOut); free(offIn); free(inPos); free(srcOut); free(dstOut); free(wOut); free(F);
        offOut = offIn = inPos = srcOut = dstOut = NULL; wOut = F = NULL;
        csr_Ncap = 0; csr_Ecap = 0;
        csr_T = 0; csr_N = 0; csr_E = 0; csr_valid = 0;
    }
    if (csr_Ncap != N) {
        free(offOut); free(offIn);
        offOut = offIn = NULL; csr_Ncap = 0;
        if (N > 0) {
            offOut = (int32_t *)malloc((size_t)(N + 1) * 4);
            offIn  = (int32_t *)malloc((size_t)(N + 1) * 4);
        }
        if (!offOut || !offIn) return 0;
        csr_Ncap = N;
    }
    if (csr_Ecap != E) {
        free(inPos); free(srcOut); free(dstOut); free(wOut); free(F);
        inPos = srcOut = dstOut = NULL; wOut = F = NULL; csr_Ecap = 0;
        if (E > 0) {
            inPos  = (int32_t *)malloc((size_t)E * 4);
            srcOut = (int32_t *)malloc((size_t)E * 4);
            dstOut = (int32_t *)malloc((size_t)E * 4);
            wOut   = (double *)malloc((size_t)E * 8);
            F      = (double *)malloc((size_t)E * 8);
        }
        if (!inPos || !srcOut || !dstOut || !wOut || !F) return 0;
        csr_Ecap = E;
    }
    if (T > 0 && N > 0 && E > 0) {
        cntOut = (int32_t *)malloc((size_t)T * (size_t)N * 4);
        cntIn  = (int32_t *)malloc((size_t)T * (size_t)N * 4);
        outIdx = (int32_t *)malloc((size_t)E * 4);
        inIdx  = (int32_t *)malloc((size_t)E * 4);
        posOfOut = (int32_t *)malloc((size_t)E * 4);
        if (!cntOut || !cntIn || !outIdx || !inIdx || !posOfOut) {
            free(cntOut); free(cntIn); free(outIdx); free(inIdx); free(posOfOut);
            cntOut = cntIn = outIdx = inIdx = posOfOut = NULL;
            return 0;
        }
    }
    csr_T = T; csr_N = N; csr_E = E;
    if (T < 1 || N < 1 || E < 1) return 0;

    memset(cntOut, 0, (size_t)T * (size_t)N * 4);
    memset(cntIn, 0, (size_t)T * (size_t)N * 4);
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        int32_t *co = cntOut + (size_t)tid * (size_t)N;
        int32_t *ci = cntIn + (size_t)tid * (size_t)N;
        #pragma omp for schedule(static, 32768)
        for (int64_t e = 0; e < E; ++e) {
            co[src[e]]++;
            ci[dst[e]]++;
        }
        #pragma omp for schedule(static, 32768)
        for (int64_t v = 0; v < N; ++v) {
            int32_t s = 0, t = 0;
            for (int k = 0; k < T; ++k) {
                s += cntOut[(size_t)k * (size_t)N + (size_t)v];
                t += cntIn[(size_t)k * (size_t)N + (size_t)v];
            }
            offOut[v] = s;
            offIn[v] = t;
        }
        #pragma omp single
        {
            /* prefix (sequential, small) */
            int32_t ao = 0, ai = 0;
            for (int64_t v = 0; v < N; ++v) {
                int32_t s = offOut[v], t = offIn[v];
                offOut[v] = ao;
                offIn[v] = ai;
                ao += s; ai += t;
            }
            offOut[N] = ao; offIn[N] = ai;
        }
        /* repurpose count slabs as positions: thread t starts at
         * offOut[v] + sum of counts of threads < t */
        #pragma omp for schedule(static, 32768)
        for (int64_t v = 0; v < N; ++v) {
            int32_t ao = offOut[v], ai = offIn[v];
            for (int k = 0; k < T; ++k) {
                int32_t co_ = cntOut[(size_t)k * (size_t)N + (size_t)v];
                int32_t ci_ = cntIn[(size_t)k * (size_t)N + (size_t)v];
                cntOut[(size_t)k * (size_t)N + (size_t)v] = ao;
                cntIn[(size_t)k * (size_t)N + (size_t)v] = ai;
                ao += co_;
                ai += ci_;
            }
        }
        #pragma omp for schedule(static, 32768) nowait
        for (int64_t e = 0; e < E; ++e) {
            int32_t *po = cntOut + (size_t)omp_get_thread_num() * (size_t)N;
            int32_t p = po[src[e]]++;
            outIdx[p] = (int32_t)e;
            posOfOut[e] = p;
        }
        #pragma omp for schedule(static, 32768)
        for (int64_t e = 0; e < E; ++e) {
            int32_t *pi = cntIn + (size_t)omp_get_thread_num() * (size_t)N;
            inIdx[pi[dst[e]]++] = (int32_t)e;
        }
        #pragma omp for schedule(static, 32768)
        for (int64_t p = 0; p < E; ++p) {
            int32_t e = outIdx[p];
            srcOut[p] = src[e];
            dstOut[p] = dst[e];
            wOut[p] = w[e];
        }
        #pragma omp for schedule(static, 32768)
        for (int64_t q = 0; q < E; ++q) inPos[q] = posOfOut[inIdx[q]];
    }
    free(cntOut); free(cntIn); free(outIdx); free(inIdx); free(posOfOut);
    cntOut = cntIn = outIdx = inIdx = posOfOut = NULL;
    k_src = src; k_dst = dst; k_w = w; k_x = x;
    k_hash = h;
    csr_valid = 1;
    return 1;
}

void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict dst,
                         const int64_t *restrict src, const double *restrict w,
                         const double *restrict x, const int64_t E, const int64_t N) {
    if (N <= 0) return;
    int mt = omp_get_max_threads();
    int T = mt;
    if (T > V7_TMAX) T = V7_TMAX;
    if (E < (int64_t)T * 200000) { T = (int)(E / 200000); if (T < 1) T = 1; }

    if (E <= 0) {
        for (int64_t i = 0; i < N; ++i) Lx[i] = 0.0;
        return;
    }
    if (T <= 1 || E >= (1LL << 31)) {
        v3_path(Lx, dst, src, w, x, E, N, T);
        return;
    }

    if (!csr_ensure(E, N, T, src, dst, w, x)) {
        v3_path(Lx, dst, src, w, x, E, N, T);
        return;
    }

    #pragma omp parallel num_threads(T)
    {
        const int64_t E16 = E & ~15LL;
        #pragma omp for schedule(static, 32768)
        for (int64_t p = 0; p < E16; p += 16) {
            __m256i d0 = _mm256_loadu_si256((const void *)(dstOut + p));
            __m256i d1 = _mm256_loadu_si256((const void *)(dstOut + p + 8));
            __m512d xd0 = _mm512_i32gather_pd(d0, x, 8);
            __m512d xd1 = _mm512_i32gather_pd(d1, x, 8);
            __m256i s0 = _mm256_loadu_si256((const void *)(srcOut + p));
            __m256i s1 = _mm256_loadu_si256((const void *)(srcOut + p + 8));
            __m512d xs0 = _mm512_i32gather_pd(s0, x, 8);
            __m512d xs1 = _mm512_i32gather_pd(s1, x, 8);
            __m512d w0 = _mm512_loadu_pd(wOut + p);
            __m512d w1 = _mm512_loadu_pd(wOut + p + 8);
            _mm512_storeu_pd(F + p, _mm512_mul_pd(w0, _mm512_sub_pd(xs0, xd0)));
            _mm512_storeu_pd(F + p + 8, _mm512_mul_pd(w1, _mm512_sub_pd(xs1, xd1)));
        }
        #pragma omp single
        for (int64_t p = E16; p < E; ++p) F[p] = wOut[p] * (x[srcOut[p]] - x[dstOut[p]]);
        #pragma omp for schedule(static, 4096)
        for (int64_t v = 0; v < N; ++v) {
            double s = 0.0;
            for (int32_t p = offOut[v]; p < offOut[v + 1]; ++p) s += F[p];
            for (int32_t q = offIn[v]; q < offIn[v + 1]; ++q) s -= F[inPos[q]];
            Lx[v] = s;
        }
    }
}
