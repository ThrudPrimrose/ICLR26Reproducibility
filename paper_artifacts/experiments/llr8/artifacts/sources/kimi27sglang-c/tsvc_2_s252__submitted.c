#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>
#include <omp.h>

extern int setenv(const char *, const char *, int);
extern long sysconf(int);

#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 84
#endif

void __attribute__((constructor)) tsvc_2_s252_init(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 1) {
        static char buf[64];
        int pos = 0;
        buf[pos++] = '0';
        buf[pos++] = '-';
        /* write n-1 decimal */
        long m = n - 1;
        char tmp[20];
        int tlen = 0;
        do { tmp[tlen++] = '0' + (m % 10); m /= 10; } while (m > 0);
        while (tlen > 0) buf[pos++] = tmp[--tlen];
        buf[pos] = '\0';
        setenv("GOMP_CPU_AFFINITY", buf, 0);
    }
}

static inline __m512d loadnt(const double *p) {
    return _mm512_castsi512_pd(_mm512_stream_load_si512((void *)p));
}

static inline void prefetchnta(const double *p) {
    _mm_prefetch((const char *)p, _MM_HINT_NTA);
}

static inline void serial_kernel(double *restrict a, const double *restrict b,
                                 const double *restrict c, int64_t n) {
    int64_t i = 0;
    const __m512d zero = _mm512_setzero_pd();
    const __m512i idx = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 15);
    __m512d prev = zero;

    for (; i + 8 <= n; i += 8) {
        __m512d bv = _mm512_loadu_pd(&b[i]);
        __m512d cv = _mm512_loadu_pd(&c[i]);
        __m512d sv = _mm512_mul_pd(bv, cv);
        __m512d shifted = _mm512_permutex2var_pd(sv, idx, prev);
        __m512d av = _mm512_add_pd(sv, shifted);
        _mm512_storeu_pd(&a[i], av);
        prev = sv;
    }

    double t;
    _mm_storeh_pd(&t, _mm512_extractf64x2_pd(prev, 3));
    for (; i < n; ++i) {
        double s = b[i] * c[i];
        a[i] = s + t;
        t = s;
    }
}

static inline void parallel_kernel_plain(double *restrict a, const double *restrict b,
                                         const double *restrict c, int64_t n) {
    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t start = (int64_t)tid * n / nt;
        int64_t end   = (int64_t)(tid + 1) * n / nt;

        double t = 0.0;
        if (start > 0) {
            t = b[start - 1] * c[start - 1];
        }

        int64_t vstart = (start + 7) & ~(int64_t)7;
        if (vstart > end) vstart = end;
        int64_t i = start;
        for (; i < vstart; ++i) {
            double s = b[i] * c[i];
            a[i] = s + t;
            t = s;
        }

        const __m512i idx = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 15);
        __m512d prev = _mm512_set1_pd(t);
        int64_t vend = end & ~(int64_t)7;
        const int64_t PF = 256;

        for (; i + 8 <= vend; i += 8) {
            prefetchnta(&b[i + PF]);
            prefetchnta(&c[i + PF]);
            __m512d bv = _mm512_loadu_pd(&b[i]);
            __m512d cv = _mm512_loadu_pd(&c[i]);
            __m512d sv = _mm512_mul_pd(bv, cv);
            __m512d shifted = _mm512_permutex2var_pd(sv, idx, prev);
            __m512d av = _mm512_add_pd(sv, shifted);
            _mm512_storeu_pd(&a[i], av);
            prev = sv;
        }

        if (i < end) {
            double tail_t;
            _mm_storeh_pd(&tail_t, _mm512_extractf64x2_pd(prev, 3));
            for (; i < end; ++i) {
                double s = b[i] * c[i];
                a[i] = s + tail_t;
                tail_t = s;
            }
        }
    }
}

static inline void parallel_kernel_nt(double *restrict a, const double *restrict b,
                                      const double *restrict c, int64_t n) {
    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t start = (int64_t)tid * n / nt;
        int64_t end   = (int64_t)(tid + 1) * n / nt;

        double t = 0.0;
        if (start > 0) {
            t = b[start - 1] * c[start - 1];
        }

        int64_t vstart = (start + 7) & ~(int64_t)7;
        if (vstart > end) vstart = end;
        int64_t i = start;
        for (; i < vstart; ++i) {
            double s = b[i] * c[i];
            a[i] = s + t;
            t = s;
        }

        const __m512i idx = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 15);
        __m512d prev = _mm512_set1_pd(t);
        int64_t vend = end & ~(int64_t)7;
        const int64_t PF = 256;

        for (; i + 8 <= vend; i += 8) {
            prefetchnta(&b[i + PF]);
            prefetchnta(&c[i + PF]);
            __m512d bv = loadnt(&b[i]);
            __m512d cv = loadnt(&c[i]);
            __m512d sv = _mm512_mul_pd(bv, cv);
            __m512d shifted = _mm512_permutex2var_pd(sv, idx, prev);
            __m512d av = _mm512_add_pd(sv, shifted);
            _mm512_stream_pd(&a[i], av);
            prev = sv;
        }
        _mm_sfence();

        if (i < end) {
            double tail_t;
            _mm_storeh_pd(&tail_t, _mm512_extractf64x2_pd(prev, 3));
            for (; i < end; ++i) {
                double s = b[i] * c[i];
                a[i] = s + tail_t;
                tail_t = s;
            }
        }
    }
}

void tsvc_2_s252_fp64(double *restrict a, double *restrict b, double *restrict c,
                      int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    if (LEN_1D <= 0) return;
    if (LEN_1D < 4096) {
        serial_kernel(a, b, c, LEN_1D);
        return;
    }
    bool aligned = (((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 63) == 0;
    if (aligned) {
        parallel_kernel_nt(a, b, c, LEN_1D);
    } else {
        parallel_kernel_plain(a, b, c, LEN_1D);
    }
}
