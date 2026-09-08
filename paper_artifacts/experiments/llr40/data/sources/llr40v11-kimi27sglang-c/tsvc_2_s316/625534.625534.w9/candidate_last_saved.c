#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

#ifdef __AVX512F__
static inline double scan_min_simd(const double *restrict a, int64_t start, int64_t end, double x) {
    __m512d v = _mm512_set1_pd(x);
    int64_t i = start;

    for (; i + 8 <= end; i += 8) {
        __m512d av = _mm512_loadu_pd(a + i);
        __mmask8 lt = _mm512_cmp_pd_mask(av, v, _CMP_LT_OQ); /* av < x, false for NaN */
        v = _mm512_mask_blend_pd(lt, v, av);                /* keep old min where false */
    }

    /* Reduce the 8 vector lanes using the exact scalar semantics. */
    _Alignas(64) double buf[8];
    _mm512_storeu_pd(buf, v);
    for (int j = 0; j < 8; ++j) {
        if (buf[j] < x) x = buf[j];
    }

    for (; i < end; ++i) {
        if (a[i] < x) x = a[i];
    }
    return x;
}
#else
static inline double scan_min_simd(const double *restrict a, int64_t start, int64_t end, double x) {
    for (int64_t i = start; i < end; ++i)
        if (a[i] < x) x = a[i];
    return x;
}
#endif

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
    double x = a[0];

    if (LEN_1D <= 1) {
        result[0] = x;
        return;
    }

    if (LEN_1D < 4096) {
        x = scan_min_simd(a, 1, LEN_1D, x);
    } else {
        int nt = 1;
        #ifdef _OPENMP
        nt = omp_get_max_threads();
        #endif
        double priv[nt];

        #pragma omp parallel
        {
            int tid = 0;
            int nthreads = 1;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
            #endif

            int64_t n = LEN_1D - 1;
            int64_t chunk = (n + nthreads - 1) / nthreads;
            int64_t lo = 1 + tid * chunk;
            int64_t hi = lo + chunk;
            if (hi > LEN_1D) hi = LEN_1D;

            double local = x; /* start from a[0] to keep NaN semantics */
            if (lo < hi) {
                local = scan_min_simd(a, lo, hi, local);
            }
            priv[tid] = local;
        }

        for (int t = 0; t < nt; ++t) {
            if (priv[t] < x) x = priv[t];
        }
    }

    result[0] = x;
}
