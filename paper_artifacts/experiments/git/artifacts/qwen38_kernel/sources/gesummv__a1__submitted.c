/* out = alpha * (A @ x) + beta * (B @ x), fused single pass.
 * Row-major A, B of shape N x N. OpenMP over rows, AVX-512 FMA inner loop. */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void gesummv_fp64(const double *restrict A, const double *restrict B,
                  double *restrict out, const double *restrict x,
                  int64_t N, double alpha, double beta) {
    #pragma omp parallel for schedule(static, 16)
    for (int64_t i = 0; i < N; ++i) {
        const double *ar = A + i * N;
        const double *br = B + i * N;
        double s = 0.0;
        int64_t j = 0;
#if defined(__AVX512F__)
        const int64_t n8 = N & ~7;
        __m512d vax = _mm512_set1_pd(alpha);
        __m512d vbx = _mm512_set1_pd(beta);
        __m512d vs = _mm512_setzero_pd();
        for (; j + 8 <= n8; j += 8) {
            __m512d va = _mm512_loadu_pd(ar + j);
            __m512d vb = _mm512_loadu_pd(br + j);
            __m512d vx = _mm512_loadu_pd(x + j);
            /* vs += (va*alpha + vb*beta) * vx */
            __m512d t = _mm512_fmadd_pd(va, vax, _mm512_mul_pd(vb, vbx));
            vs = _mm512_fmadd_pd(t, vx, vs);
        }
        s = _mm512_reduce_add_pd(vs);
#endif
        for (; j < N; ++j) {
            s += (ar[j] * alpha + br[j] * beta) * x[j];
        }
        out[i] = s;
    }
}
