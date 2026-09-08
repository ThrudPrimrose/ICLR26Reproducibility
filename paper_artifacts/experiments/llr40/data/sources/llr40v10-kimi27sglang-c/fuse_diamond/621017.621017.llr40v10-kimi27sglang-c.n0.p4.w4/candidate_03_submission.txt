#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

__attribute__((always_inline, optimize("no-stack-protector")))
static inline void fuse_diamond_seq(const double *restrict a, double *restrict out, const int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        double t = a[i] * a[i];
        out[i] = (t + 1.0) * (t - 1.0);
    }
}

__attribute__((optimize("no-stack-protector")))
void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    if (LEN_1D < 4096) {
        fuse_diamond_seq(a, out, LEN_1D);
        return;
    }
    if ((((intptr_t)a | (intptr_t)out) & 63) == 0) {
        const __m512d vone = _mm512_set1_pd(1.0);
        const __m512d vmone = _mm512_set1_pd(-1.0);
        const int64_t main = LEN_1D & ~7;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < main; i += 8) {
            __m512d va = _mm512_load_pd(a + i);
            __m512d t = _mm512_mul_pd(va, va);
            __m512d u = _mm512_fmadd_pd(t, vone, vone);  /* t + 1 */
            __m512d v = _mm512_fmadd_pd(t, vone, vmone); /* t - 1 */
            _mm512_stream_pd(out + i, _mm512_mul_pd(u, v));
        }
        _mm_sfence();
        for (int64_t i = main; i < LEN_1D; ++i) {
            double t = a[i] * a[i];
            out[i] = (t + 1.0) * (t - 1.0);
        }
    } else {
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double t = a[i] * a[i];
            out[i] = (t + 1.0) * (t - 1.0);
        }
    }
}
