#include <stdint.h>
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

static inline int64_t process_chunk(double *restrict a, const double *restrict b, const double *restrict c, int64_t n) {
    int64_t i = 0;

    if (n >= 8) {
        const __m512i idx = _mm512_setr_epi64(8, 0, 1, 2, 3, 4, 5, 6);
        const __m512i broadcast7 = _mm512_set1_epi64(7);
        __m512d prev_bcast = _mm512_set1_pd(0.0);

        for (; i + 8 <= n; i += 8) {
            __m512d bv = _mm512_loadu_pd(&b[i]);
            __m512d cv = _mm512_loadu_pd(&c[i]);
            __m512d sv = _mm512_mul_pd(bv, cv);

            __m512d shifted = _mm512_permutex2var_pd(sv, idx, prev_bcast);
            __m512d av = _mm512_add_pd(sv, shifted);
            _mm512_storeu_pd(&a[i], av);

            prev_bcast = _mm512_permutexvar_pd(broadcast7, sv);
        }
    }

    double prev = 0.0;
    for (; i < n; ++i) {
        double s = b[i] * c[i];
        a[i] = s + prev;
        prev = s;
    }

    return i > 0 ? (int64_t)1 : (int64_t)0;
}

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    // Use single-threaded path for small inputs to avoid OpenMP overhead
    if (LEN_1D < 4096) {
        process_chunk(a, b, c, LEN_1D);
        return;
    }

    int nthreads = omp_get_max_threads();
    if (nthreads <= 1) {
        process_chunk(a, b, c, LEN_1D);
        return;
    }

    int64_t chunk = (LEN_1D + nthreads - 1) / nthreads;
    double *last_s = (double *)malloc(nthreads * sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (start < LEN_1D) {
            if (end > LEN_1D) end = LEN_1D;
            int64_t n = end - start;
            process_chunk(a + start, b + start, c + start, n);
            last_s[tid] = b[end - 1] * c[end - 1];
        } else {
            last_s[tid] = 0.0;
        }
    }

    // Fix chunk boundaries serially
    for (int j = 1; j < nthreads; ++j) {
        int64_t start = (int64_t)j * chunk;
        if (start < LEN_1D) {
            a[start] += last_s[j - 1];
        }
    }

    free(last_s);
}
