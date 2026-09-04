#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    const int64_t M = N - 1;
    #pragma omp parallel for ordered schedule(static,1)
    for (int64_t i = 1; i < N; ++i) {
        #pragma omp ordered depend(sink: i-1)
        double *restrict cur = a + i * N;
        const double *restrict prev = a + (i - 1) * N;
        int64_t j = 0;
        for (; j + 8 <= M; j += 8) {
            __m512d c = _mm512_loadu_pd(cur + j);
            __m512d p = _mm512_loadu_pd(prev + j);
            __m512d p1 = _mm512_loadu_pd(prev + j + 1);
            __m512d s = _mm512_add_pd(c, p);
            s = _mm512_add_pd(s, p1);
            _mm512_storeu_pd(cur + j, s);
        }
        for (; j < M; ++j) {
            cur[j] = cur[j] + prev[j] + prev[j + 1];
        }
        #pragma omp ordered depend(source)
    }
}
