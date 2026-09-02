#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline double reduce_positive_avx512(const double * restrict a, int64_t n)
{
    const __m512d zero = _mm512_setzero_pd();
    const int64_t n32 = n & ~(int64_t)31;

    __m512d v0 = _mm512_setzero_pd();
    __m512d v1 = _mm512_setzero_pd();
    __m512d v2 = _mm512_setzero_pd();
    __m512d v3 = _mm512_setzero_pd();

    int64_t i;
    for (i = 0; i < n32; i += 32) {
        __m512d x0 = _mm512_loadu_pd(a + i + 0);
        __m512d x1 = _mm512_loadu_pd(a + i + 8);
        __m512d x2 = _mm512_loadu_pd(a + i + 16);
        __m512d x3 = _mm512_loadu_pd(a + i + 24);
        v0 = _mm512_add_pd(v0, _mm512_max_pd(x0, zero));
        v1 = _mm512_add_pd(v1, _mm512_max_pd(x1, zero));
        v2 = _mm512_add_pd(v2, _mm512_max_pd(x2, zero));
        v3 = _mm512_add_pd(v3, _mm512_max_pd(x3, zero));
    }

    __m512d vt = _mm512_add_pd(_mm512_add_pd(v0, v1), _mm512_add_pd(v2, v3));
    double sum = _mm512_reduce_add_pd(vt);

    for (; i < n; ++i) {
        if (a[i] > 0.0) sum += a[i];
    }
    return sum;
}

void tsvc_2_s3111_fp64(double * restrict a, double * restrict b, int64_t LEN_1D,
                       uint8_t * restrict workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;

    double sum;
    if (LEN_1D < 4096) {
        sum = reduce_positive_avx512(a, LEN_1D);
    } else {
        const int64_t n32 = LEN_1D & ~(int64_t)31;
        double partial[64] = {0.0};

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            __m512d v0 = _mm512_setzero_pd();
            __m512d v1 = _mm512_setzero_pd();
            __m512d v2 = _mm512_setzero_pd();
            __m512d v3 = _mm512_setzero_pd();
            const __m512d zero = _mm512_setzero_pd();

            int64_t i;
            #pragma omp for schedule(static) nowait
            for (i = 0; i < n32; i += 32) {
                __m512d x0 = _mm512_loadu_pd(a + i + 0);
                __m512d x1 = _mm512_loadu_pd(a + i + 8);
                __m512d x2 = _mm512_loadu_pd(a + i + 16);
                __m512d x3 = _mm512_loadu_pd(a + i + 24);
                v0 = _mm512_add_pd(v0, _mm512_max_pd(x0, zero));
                v1 = _mm512_add_pd(v1, _mm512_max_pd(x1, zero));
                v2 = _mm512_add_pd(v2, _mm512_max_pd(x2, zero));
                v3 = _mm512_add_pd(v3, _mm512_max_pd(x3, zero));
            }

            __m512d vt = _mm512_add_pd(_mm512_add_pd(v0, v1), _mm512_add_pd(v2, v3));
            partial[tid] = _mm512_reduce_add_pd(vt);
        }

        sum = 0.0;
        int nthreads = omp_get_max_threads();
        for (int t = 0; t < nthreads; ++t) sum += partial[t];

        for (int64_t i = n32; i < LEN_1D; ++i) {
            if (a[i] > 0.0) sum += a[i];
        }
    }

    b[0] = sum;
}
