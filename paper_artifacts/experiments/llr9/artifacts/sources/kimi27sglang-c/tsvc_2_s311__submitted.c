#include <stdint.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 192
#endif

static inline double reduce_chunk(const double *restrict a, int64_t n) {
    double sum = 0.0;
    int64_t i = 0;

    for (; i < n && (((uintptr_t)(a + i)) & 63) != 0; ++i) {
        sum += a[i];
    }

    __m512d s0 = _mm512_setzero_pd();
    __m512d s1 = _mm512_setzero_pd();

    for (; i + 15 < n; i += 16) {
        s0 = _mm512_add_pd(s0, _mm512_load_pd(a + i));
        s1 = _mm512_add_pd(s1, _mm512_load_pd(a + i + 8));
    }

    for (; i + 7 < n; i += 8) {
        s0 = _mm512_add_pd(s0, _mm512_load_pd(a + i));
    }

    sum += _mm512_reduce_add_pd(_mm512_add_pd(s0, s1));

    for (; i < n; ++i) {
        sum += a[i];
    }

    return sum;
}

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
    double sum = 0.0;

    if (LEN_1D < 4096) {
        sum = reduce_chunk(a, LEN_1D);
    } else {
#ifdef _OPENMP
        double partial[MAX_THREADS];
        int actual_threads = 0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            if (tid == 0) {
                actual_threads = nt;
            }
            int64_t chunk = LEN_1D / nt;
            int64_t start = (int64_t)tid * chunk;
            int64_t end = (tid == nt - 1) ? LEN_1D : start + chunk;
            partial[tid] = reduce_chunk(a + start, end - start);
        }

        for (int t = 0; t < actual_threads; ++t) {
            sum += partial[t];
        }
#else
        sum = reduce_chunk(a, LEN_1D);
#endif
    }

    sum_out[0] = sum;
}
