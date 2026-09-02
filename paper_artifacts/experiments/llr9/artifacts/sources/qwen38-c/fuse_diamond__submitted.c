#include <immintrin.h>
#include <stdint.h>

#pragma GCC optimize("fp-contract=off")
void fuse_diamond_fp64(double *restrict a, double *restrict out, int64_t LEN_1D,
                       uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace; (void)workspace_bytes;
    int64_t n8 = LEN_1D / 8;
    if (((uintptr_t)a & 63) == 0 && ((uintptr_t)out & 63) == 0) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int64_t k = 0; k < n8; k++) {
                int64_t i = k << 3;
                __m512d x = _mm512_load_pd(a + i);
                __m512d one = _mm512_set1_pd(1.0);
                __m512d t = _mm512_mul_pd(x, x);
                __m512d u = _mm512_add_pd(t, one);
                __m512d v = _mm512_sub_pd(t, one);
                _mm512_stream_pd(out + i, _mm512_mul_pd(u, v));
            }
            _mm_sfence();
        }
        for (int64_t i = n8*8; i < LEN_1D; i++) {
            double t = a[i]*a[i];
            double u = t+1.0;
            double v = t-1.0;
            out[i] = u*v;
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_1D; i++) {
            double t = a[i]*a[i];
            double u = t+1.0;
            double v = t-1.0;
            out[i] = u*v;
        }
    }
}
