#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#define MAX_THREADS 256
#define MIN_CHUNK 8192
#define PARALLEL_THRESH (2 * MIN_CHUNK)

static inline __m512d scan_vec(__m512d v, __m512d sum, __m512i zero_si) {
    __m512d prev = v;
    v = _mm512_add_pd(v, _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(prev), zero_si, 7)));
    prev = v;
    v = _mm512_add_pd(v, _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(prev), zero_si, 6)));
    prev = v;
    v = _mm512_add_pd(v, _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(prev), zero_si, 4)));
    return _mm512_add_pd(v, sum);
}

static inline __m512d broadcast_last(__m512d v, __m512i idx7) {
    return _mm512_permutexvar_pd(idx7, v);
}

static inline void store512_pd(double *p, __m512d v, int stream) {
    if (stream) _mm512_stream_pd(p, v);
    else _mm512_storeu_pd(p, v);
}

static void scan_sequential(const double *restrict a, double *restrict b, int64_t n) {
    int64_t i = 0;
    const __m512i zero_si = _mm512_setzero_si512();
    __m512d sum = _mm512_setzero_pd();
    const __m512i idx7 = _mm512_set1_epi64(7);

    for (; i + 32 <= n; i += 32) {
        __m512d v0 = _mm512_loadu_pd(&a[i]);
        __m512d v1 = _mm512_loadu_pd(&a[i + 8]);
        __m512d v2 = _mm512_loadu_pd(&a[i + 16]);
        __m512d v3 = _mm512_loadu_pd(&a[i + 24]);

        v0 = scan_vec(v0, sum, zero_si);
        sum = broadcast_last(v0, idx7);
        v1 = scan_vec(v1, sum, zero_si);
        sum = broadcast_last(v1, idx7);
        v2 = scan_vec(v2, sum, zero_si);
        sum = broadcast_last(v2, idx7);
        v3 = scan_vec(v3, sum, zero_si);
        sum = broadcast_last(v3, idx7);

        _mm512_storeu_pd(&b[i], v0);
        _mm512_storeu_pd(&b[i + 8], v1);
        _mm512_storeu_pd(&b[i + 16], v2);
        _mm512_storeu_pd(&b[i + 24], v3);
    }
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(&a[i]);
        v = scan_vec(v, sum, zero_si);
        _mm512_storeu_pd(&b[i], v);
        sum = broadcast_last(v, idx7);
    }

    double s = _mm512_cvtsd_f64(sum);
    for (; i < n; ++i) {
        s += a[i];
        b[i] = s;
    }
}

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D < PARALLEL_THRESH) {
        scan_sequential(a, b, LEN_1D);
        return;
    }

    int stream = (((uintptr_t)b & 63) == 0);

    int max_threads = omp_get_max_threads();
    int nt = max_threads;
    int64_t desired = (LEN_1D + MIN_CHUNK - 1) / MIN_CHUNK;
    if (desired < nt) nt = (int)desired;
    if (nt < 1) nt = 1;
    if (nt > MAX_THREADS) nt = MAX_THREADS;

    __attribute__((aligned(64))) double totals[MAX_THREADS] = {0};
    __attribute__((aligned(64))) double offsets[MAX_THREADS] = {0};

    int64_t chunk = ((LEN_1D + nt - 1) / nt + 7) & ~7LL;
    const __m512i zero_si = _mm512_setzero_si512();
    const __m512i idx7 = _mm512_set1_epi64(7);

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;

        // Phase 1: local reduction
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        int64_t i = start;
        for (; i + 32 <= end; i += 32) {
            s0 = _mm512_add_pd(s0, _mm512_loadu_pd(&a[i]));
            s1 = _mm512_add_pd(s1, _mm512_loadu_pd(&a[i + 8]));
            s2 = _mm512_add_pd(s2, _mm512_loadu_pd(&a[i + 16]));
            s3 = _mm512_add_pd(s3, _mm512_loadu_pd(&a[i + 24]));
        }
        for (; i + 8 <= end; i += 8) {
            s0 = _mm512_add_pd(s0, _mm512_loadu_pd(&a[i]));
        }
        s0 = _mm512_add_pd(s0, s1);
        s2 = _mm512_add_pd(s2, s3);
        s0 = _mm512_add_pd(s0, s2);
        double t = _mm512_reduce_add_pd(s0);
        for (; i < end; ++i) t += a[i];
        totals[tid] = t;

        #pragma omp barrier
        #pragma omp single
        {
            offsets[0] = 0.0;
            for (int tt = 1; tt < nt; ++tt) {
                offsets[tt] = offsets[tt - 1] + totals[tt - 1];
            }
        }
        #pragma omp barrier

        // Phase 2: local scan with chunk offset
        __m512d sum = _mm512_set1_pd(offsets[tid]);
        i = start;
        for (; i + 32 <= end; i += 32) {
            __m512d v0 = _mm512_loadu_pd(&a[i]);
            __m512d v1 = _mm512_loadu_pd(&a[i + 8]);
            __m512d v2 = _mm512_loadu_pd(&a[i + 16]);
            __m512d v3 = _mm512_loadu_pd(&a[i + 24]);

            v0 = scan_vec(v0, sum, zero_si);
            sum = broadcast_last(v0, idx7);
            v1 = scan_vec(v1, sum, zero_si);
            sum = broadcast_last(v1, idx7);
            v2 = scan_vec(v2, sum, zero_si);
            sum = broadcast_last(v2, idx7);
            v3 = scan_vec(v3, sum, zero_si);
            sum = broadcast_last(v3, idx7);

            store512_pd(&b[i], v0, stream);
            store512_pd(&b[i + 8], v1, stream);
            store512_pd(&b[i + 16], v2, stream);
            store512_pd(&b[i + 24], v3, stream);
        }
        for (; i + 8 <= end; i += 8) {
            __m512d v = _mm512_loadu_pd(&a[i]);
            v = scan_vec(v, sum, zero_si);
            store512_pd(&b[i], v, stream);
            sum = broadcast_last(v, idx7);
        }
        double s = _mm512_cvtsd_f64(sum);
        for (; i < end; ++i) {
            s += a[i];
            b[i] = s;
        }
    }
    if (stream) _mm_sfence();
}
