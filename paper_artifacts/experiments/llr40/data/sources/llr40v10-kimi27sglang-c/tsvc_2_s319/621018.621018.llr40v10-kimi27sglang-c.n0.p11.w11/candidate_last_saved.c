#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    const int64_t vec = 8;
    const int64_t n_vec = LEN_1D / vec;
    double sum = 0.0;

    #pragma omp parallel reduction(+:sum)
    {
        __m512d vsum0 = _mm512_setzero_pd();
        __m512d vsum1 = _mm512_setzero_pd();

        #pragma omp for nowait schedule(static)
        for (int64_t j = 0; j < n_vec; ++j) {
            int64_t i = j * vec;
            __m512d ci = _mm512_loadu_pd(&c[i]);
            __m512d di = _mm512_loadu_pd(&d[i]);
            __m512d ei = _mm512_loadu_pd(&e[i]);
            __m512d ai = _mm512_add_pd(ci, di);
            __m512d bi = _mm512_add_pd(ci, ei);
            _mm512_storeu_pd(&a[i], ai);
            _mm512_storeu_pd(&b[i], bi);
            vsum0 = _mm512_add_pd(vsum0, ai);
            vsum1 = _mm512_add_pd(vsum1, bi);
        }

        double local_sum = _mm512_reduce_add_pd(_mm512_add_pd(vsum0, vsum1));
        sum += local_sum;
    }

    for (int64_t i = n_vec * vec; i < LEN_1D; ++i) {
        a[i] = c[i] + d[i];
        sum += a[i];
        b[i] = c[i] + e[i];
        sum += b[i];
    }

    b[0] = sum;
}
