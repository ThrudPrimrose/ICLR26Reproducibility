#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <emmintrin.h>

#ifndef MAX_K
#define MAX_K 512
#endif

#ifndef BTARGET
#define BTARGET 32768
#endif

static inline void stream_store_double(double *p, double v)
{
    union { double d; long long i; } u;
    u.d = v;
    _mm_stream_si64((long long *)p, u.i);
}

static inline void run_k1_ws(double *restrict a, const double *restrict b,
                             const double *restrict c, int64_t N,
                             double *restrict d)
{
    if (N <= 1) return;
    const double alpha = 0.75;
    int T = omp_get_max_threads();
    if (T > N / 2) T = (int)(N / 2);
    if (T < 1) T = 1;
    double aggA[256];
    double aggB[256];
    double seeds[256];
    int64_t block = (N - 1 + T - 1) / T;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t s = 1 + (int64_t)tid * block;
        int64_t e = s + block;
        if (e > N) e = N;
        double A = 1.0, B = 0.0;
        #pragma omp simd
        for (int64_t i = s; i < e; ++i) {
            double di = b[i] * c[i];
            d[i] = di;
            A *= alpha;
            B = alpha * B + di;
        }
        aggA[tid] = A;
        aggB[tid] = B;
    }

    double seed = a[0];
    seeds[0] = seed;
    for (int t = 0; t < T - 1; ++t) {
        seed = aggA[t] * seed + aggB[t];
        seeds[t + 1] = seed;
    }

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        int64_t s = 1 + (int64_t)t * block;
        int64_t e = s + block;
        if (e > N) e = N;
        double x = seeds[t];
        for (int64_t i = s; i < e; ++i) {
            x = alpha * x + d[i];
            stream_store_double(&a[i], x);
        }
        _mm_sfence();
    }
}

static inline void run_k1_nows(double *restrict a, const double *restrict b,
                               const double *restrict c, int64_t N)
{
    if (N <= 1) return;
    const double alpha = 0.75;
    int T = omp_get_max_threads();
    if (T > N / 2) T = (int)(N / 2);
    if (T < 1) T = 1;
    double aggA[256];
    double aggB[256];
    double seeds[256];
    int64_t block = (N - 1 + T - 1) / T;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t s = 1 + (int64_t)tid * block;
        int64_t e = s + block;
        if (e > N) e = N;
        double A = 1.0, B = 0.0;
        #pragma omp simd
        for (int64_t i = s; i < e; ++i) {
            double di = b[i] * c[i];
            A *= alpha;
            B = alpha * B + di;
        }
        aggA[tid] = A;
        aggB[tid] = B;
    }

    double seed = a[0];
    seeds[0] = seed;
    for (int t = 0; t < T - 1; ++t) {
        seed = aggA[t] * seed + aggB[t];
        seeds[t + 1] = seed;
    }

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        int64_t s = 1 + (int64_t)t * block;
        int64_t e = s + block;
        if (e > N) e = N;
        double x = seeds[t];
        for (int64_t i = s; i < e; ++i) {
            x = alpha * x + b[i] * c[i];
            stream_store_double(&a[i], x);
        }
        _mm_sfence();
    }
}

static inline void run_block_serial(double *restrict a, const double *restrict b,
                                    const double *restrict c, int64_t K, int64_t N)
{
    if (K <= 0 || N <= K) return;
    const double alpha = 0.75;
    double prev[MAX_K];
    for (int64_t r = 0; r < K; ++r) prev[r] = a[r];
    int64_t M = N / K;
    for (int64_t m = 0; m < M; ++m) {
        int64_t base = m * K;
        for (int64_t r = 0; r < K; ++r) {
            int64_t i = base + r;
            double x = alpha * prev[r] + b[i] * c[i];
            a[i] = x;
            prev[r] = x;
        }
    }
    int64_t rem = N - M * K;
    int64_t base = M * K;
    for (int64_t r = 0; r < rem; ++r) {
        int64_t i = base + r;
        double x = alpha * prev[r] + b[i] * c[i];
        a[i] = x;
    }
}

static inline void run_block_scan(double *restrict a, const double *restrict b,
                                  const double *restrict c, int64_t K, int64_t N,
                                  double *restrict work, int64_t work_bytes)
{
    if (K <= 0 || N <= K) return;
    const double alpha = 0.75;

    int64_t B = (BTARGET / K) * K;
    if (B < K) B = K;
    int64_t L = B / K;
    double A = pow(alpha, (double)L);
    int64_t start = K;
    int64_t len = N - start;
    int64_t T = len / B;
    int64_t rem = len - T * B;
    if (T <= 0) {
        run_block_serial(a, b, c, K, N);
        return;
    }

    int64_t need1 = T * K * sizeof(double);
    if ((int64_t)need1 > work_bytes) {
        run_block_serial(a, b, c, K, N);
        return;
    }
    double *Bfinal = work;

    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; ++t) {
        int64_t s = start + t * B;
        double prev[MAX_K];
        for (int64_t r = 0; r < K; ++r) prev[r] = 0.0;
        for (int64_t l = 0; l < L; ++l) {
            int64_t base = s + l * K;
            #pragma omp simd
            for (int64_t r = 0; r < K; ++r) {
                int64_t i = base + r;
                prev[r] = alpha * prev[r] + b[i] * c[i];
            }
        }
        for (int64_t r = 0; r < K; ++r) Bfinal[t * K + r] = prev[r];
    }

    uintptr_t p2 = ((uintptr_t)(char *)work + need1 + 63) & ~(uintptr_t)63;
    double *block_seeds = (double *)p2;
    int64_t need2 = T * K * sizeof(double);
    if ((char *)block_seeds - (char *)work + need2 > work_bytes) {
        run_block_serial(a, b, c, K, N);
        return;
    }
    double prev_seed[MAX_K];
    for (int64_t r = 0; r < K; ++r) prev_seed[r] = a[r];
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t r = 0; r < K; ++r) {
            block_seeds[t * K + r] = prev_seed[r];
            prev_seed[r] = A * prev_seed[r] + Bfinal[t * K + r];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; ++t) {
        int64_t s = start + t * B;
        double prev[MAX_K];
        for (int64_t r = 0; r < K; ++r) prev[r] = block_seeds[t * K + r];
        for (int64_t l = 0; l < L; ++l) {
            int64_t base = s + l * K;
            #pragma omp simd
            for (int64_t r = 0; r < K; ++r) {
                int64_t i = base + r;
                prev[r] = alpha * prev[r] + b[i] * c[i];
                a[i] = prev[r];
            }
        }
    }

    if (rem > 0) {
        int64_t s = start + T * B;
        double prev[MAX_K];
        for (int64_t r = 0; r < K; ++r) prev[r] = prev_seed[r];
        for (int64_t i = s; i < N; ++i) {
            int64_t r = (i - start) % K;
            prev[r] = alpha * prev[r] + b[i] * c[i];
            a[i] = prev[r];
        }
    }
}

void versioned_distance_update_fp64(double *restrict a, double *restrict b,
                                    double *restrict c,
                                    int64_t K, int64_t LEN_1D,
                                    uint8_t *workspace, int64_t workspace_bytes)
{
    if (K <= 0 || LEN_1D <= K) return;

    double *work = NULL;
    int64_t work_avail = 0;
    if (workspace != NULL && workspace_bytes >= 64) {
        uintptr_t p = ((uintptr_t)workspace + 63) & ~(uintptr_t)63;
        work = (double *)p;
        work_avail = workspace_bytes - (int64_t)(p - (uintptr_t)workspace);
        if (work_avail < 0) work_avail = 0;
    }

    if (K == 1) {
        int64_t need_d = LEN_1D * sizeof(double);
        if (work != NULL && work_avail >= need_d)
            run_k1_ws(a, b, c, LEN_1D, work);
        else
            run_k1_nows(a, b, c, LEN_1D);
    } else {
        run_block_scan(a, b, c, K, LEN_1D, work, work_avail);
    }
}
