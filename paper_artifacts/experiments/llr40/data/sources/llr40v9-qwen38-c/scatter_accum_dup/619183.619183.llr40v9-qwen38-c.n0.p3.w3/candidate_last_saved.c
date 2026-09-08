#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

/* ------------------------------------------------------------------ */
/* Atomic fallback (v1): correct for any N / missing workspace.       */
/* ------------------------------------------------------------------ */
static inline void sad_dadd(double *p, double v)
{
    __asm__ volatile(
        ".__sad%=: movq (%[p]), %%rax\n\t"
        "     movq      %%rax, %%xmm0\n\t"
        "     addsd     %[v], %%xmm0\n\t"
        "     movq      %%xmm0, %%rdx\n\t"
        "     lock cmpxchgq %%rdx, (%[p])\n\t"
        "     jnz       .__sad%=\n\t"
        :
        : [v] "xm"(v), [p] "r"(p)
        : "rax", "rdx", "xmm0", "memory", "cc"
    );
}

static void sad_atomic_all(double *bins, const int32_t *ip, const double *src, int64_t n)
{
#pragma omp parallel
    {
        int64_t nt  = omp_get_num_threads();
        int64_t tid = omp_get_thread_num();
        int64_t chunk = (n + nt - 1) / nt;
        int64_t beg = tid * chunk;
        int64_t end = beg + chunk;
        if (end > n) end = n;
        int64_t i = beg;
        int64_t lim = beg + ((end - beg) & ~3);
        for (; i < lim; i += 4) {
            int32_t j0 = ip[i], j1 = ip[i+1], j2 = ip[i+2], j3 = ip[i+3];
            sad_dadd(&bins[j0], src[i]);
            sad_dadd(&bins[j1], src[i+1]);
            sad_dadd(&bins[j2], src[i+2]);
            sad_dadd(&bins[j3], src[i+3]);
        }
        for (; i < end; i++)
            sad_dadd(&bins[ip[i]], src[i]);
    }
}

/* ------------------------------------------------------------------ */
/* One pass of stable LSD counting sort on (key, val) pairs.          */
/* key field = (in_k[i] >> sh) & (nb-1); threads own disjoint output  */
/* blocks per bucket -> no atomics anywhere.                          */
/* ------------------------------------------------------------------ */
#define MAXT 32

static void sad_pass(const int32_t *in_k, const double *in_v,
                     int32_t *out_k, double *out_v,
                     uint32_t *cnt, int64_t *off,
                     int64_t n, int nt, int sh, uint32_t nb)
{
    memset(cnt, 0, (size_t)nt * nb * sizeof(uint32_t));

#pragma omp parallel num_threads(nt)
    {
        int64_t chunk = (n + nt - 1) / nt;
        int64_t beg   = (int64_t)omp_get_thread_num() * chunk;
        int64_t end   = beg + chunk;
        if (end > n) end = n;
        uint32_t *mycnt = cnt + (size_t)omp_get_thread_num() * nb;
        for (int64_t i = beg; i < end; i++)
            mycnt[(in_k[i] >> sh) & (nb - 1)]++;
    }

    /* serial 2-D prefix.  Output layout is (bucket b, thread t, order), so a
     * block's start = B[b] (all earlier buckets, all threads) + T(t, b)
     * (earlier threads in the same bucket). */
    int64_t *B = malloc(nb * sizeof(int64_t));
    int64_t *run = malloc(nb * sizeof(int64_t));
    memset(B, 0, nb * sizeof(int64_t));
    for (int t = 0; t < nt; t++) {
        const uint32_t *rc = cnt + (size_t)t * nb;
        for (uint32_t b = 0; b < nb; b++)
            B[b] += rc[b];
    }
    int64_t acc = 0;
    for (uint32_t b = 0; b < nb; b++) {
        int64_t tmp = B[b];
        B[b] = acc;
        acc += tmp;
    }
    memset(run, 0, nb * sizeof(int64_t));
    for (int t = 0; t < nt; t++) {
        const uint32_t *rc = cnt + (size_t)t * nb;
        int64_t *ro = off + (size_t)t * nb;
        for (uint32_t b = 0; b < nb; b++) {
            ro[b] = B[b] + run[b];
            run[b] += rc[b];
        }
    }
    free(B);
    free(run);

    /* scatter: offsets reused as private cursors */
#pragma omp parallel num_threads(nt)
    {
        int64_t chunk = (n + nt - 1) / nt;
        int64_t beg   = (int64_t)omp_get_thread_num() * chunk;
        int64_t end   = beg + chunk;
        if (end > n) end = n;
        int t = omp_get_thread_num();
        int64_t *cur = off + (size_t)t * nb;
        for (int64_t i = beg; i < end; i++) {
            int64_t pos = cur[(in_k[i] >> sh) & (nb - 1)]++;
            out_k[pos] = in_k[i];
            out_v[pos] = in_v[i];
        }
    }
}

void scatter_accum_dup_fp64(double *bins, int32_t *ip, double *src,
                            int64_t len_1d, uint8_t *workspace,
                            int64_t workspace_size)
{
    const int64_t n = len_1d;
    if (n < 1024) {            /* tiny: just do it serially */
        for (int64_t i = 0; i < n; i++) bins[ip[i]] += src[i];
        return;
    }

    int nt = (int)omp_get_max_threads();
    if (nt < 1) nt = 1;
    if (nt > MAXT) nt = MAXT;

    const uint32_t B0 = 16384; /* pass 0: low 14 bits */
    const uint32_t B1 = 8192;  /* pass 1: bits 14..26 (needs n < 2^27) */
    int64_t need = 24 * n + (int64_t)nt * B0 * sizeof(uint32_t)   /* cnt   */
                 + (int64_t)nt * B0 * sizeof(int64_t);            /* off   */
    if (n < (1LL << 27) && workspace && workspace_size >= need) {
        int32_t *A_k = (int32_t *)(workspace);
        double  *A_v = (double  *)(workspace + 4 * n);
        int32_t *B_k = (int32_t *)(workspace + 12 * n);
        double  *B_v = (double  *)(workspace + 16 * n);
        uint8_t *meta = workspace + 24 * n;
        uint32_t *cnt = (uint32_t *)meta;
        int64_t *off  = (int64_t *)(meta + (size_t)nt * B0 * sizeof(uint32_t));

        /* pass 0: ip/src -> A, key = low 14 bits */
        sad_pass(ip, src, A_k, A_v, cnt, off, n, nt, 0, B0);
        /* pass 1: A -> B, key = bits 14..26 */
        sad_pass(A_k, A_v, B_k, B_v, cnt, off, n, nt, 14, B1);

        /* thread boundaries snapped to key-run edges: no bin shared between
         * threads -> plain (non-atomic) writes in the gather. */
        int64_t bnd[MAXT + 1];
        bnd[0] = 0;
        for (int t = 1; t < nt; t++) {
            bnd[t] = (int64_t)((int64_t)t * n / nt);
            int32_t j = B_k[bnd[t] - 1];
            while (bnd[t] < n && B_k[bnd[t]] == j) bnd[t]++;
        }
        bnd[nt] = n;

#pragma omp parallel num_threads(nt)
        {
            int t = omp_get_thread_num();
            int64_t beg = bnd[t], end = bnd[t + 1];
            double acc = 0.0;
            int cur = -1;
            for (int64_t i = beg; i < end; i++) {
                int j = B_k[i];
                if (j != cur) {
                    if (cur >= 0) bins[cur] += acc;
                    cur = j;
                    acc = 0.0;
                }
                acc += B_v[i];
            }
            if (cur >= 0) bins[cur] += acc;
        }
        return;
    }

    sad_atomic_all(bins, ip, src, n);
}
