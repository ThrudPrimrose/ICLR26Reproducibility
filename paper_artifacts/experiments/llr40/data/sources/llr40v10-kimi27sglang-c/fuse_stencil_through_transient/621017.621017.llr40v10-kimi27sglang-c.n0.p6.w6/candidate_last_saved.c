#include <stdint.h>
#include <immintrin.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    const int64_t n = LEN_1D - 2;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n - 7; i += 8) {
        __m512i v0 = _mm512_castpd_si512(_mm512_loadu_pd(a + i - 1));
        __m512i v1 = _mm512_castpd_si512(_mm512_loadu_pd(a + i + 2));
        __m512d v_im1 = _mm512_castsi512_pd(v0);
        __m512d v_i   = _mm512_castsi512_pd(_mm512_alignr_epi64(v1, v0, 1));
        __m512d v_ip1 = _mm512_castsi512_pd(_mm512_alignr_epi64(v1, v0, 2));
        __m512d v_ip2 = _mm512_castsi512_pd(_mm512_alignr_epi64(v1, v0, 3));
        __m512d t = _mm512_add_pd(v_i, v_ip1);
        __m512d L = _mm512_add_pd(v_im1, t);
        __m512d R = _mm512_add_pd(t, v_ip2);
        _mm512_storeu_pd(out + i, _mm512_mul_pd(L, R));
    }

    int64_t tail_start = (n > 8) ? (((n - 9) / 8) * 8 + 9) : 1;
    for (int64_t i = tail_start; i < n; ++i) {
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
}
