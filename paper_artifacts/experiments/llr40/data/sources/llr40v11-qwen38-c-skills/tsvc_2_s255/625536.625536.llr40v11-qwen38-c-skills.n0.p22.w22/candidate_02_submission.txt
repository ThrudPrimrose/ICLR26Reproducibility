/* TSVC tsvc_2 s255: the x/y carry in the reference is exactly b[i-1]/b[i-2],
 * so a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333 with b[-1] = b[n-1], b[-2] = b[n-2].
 * Fully parallel and vectorizable. Small n: serial AVX-512 (team fork would cost
 * more than the work); large n: parallel + simd, memory-bandwidth bound. */

#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

static void small_path(double *restrict a, const double *restrict b, const int64_t n) {
  const __m512d c = _mm512_set1_pd(0.333);
  const int64_t n4 = (n >= 34) ? (n - 34) / 32 + 1 : 0;
  for (int64_t k = 0; k < n4; k++) {
    const int64_t j = 2 + 32 * k;
    __m512d v1 = _mm512_loadu_pd(b+j),   v2 = _mm512_loadu_pd(b+j-1),  v3 = _mm512_loadu_pd(b+j-2);
    __m512d w1 = _mm512_loadu_pd(b+j+8), w2 = _mm512_loadu_pd(b+j+7),  w3 = _mm512_loadu_pd(b+j+6);
    __m512d x1 = _mm512_loadu_pd(b+j+16),x2 = _mm512_loadu_pd(b+j+15), x3 = _mm512_loadu_pd(b+j+14);
    __m512d y1 = _mm512_loadu_pd(b+j+24),y2 = _mm512_loadu_pd(b+j+23), y3 = _mm512_loadu_pd(b+j+22);
    _mm512_storeu_pd(a+j,   _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v1,v2),v3), c));
    _mm512_storeu_pd(a+j+8, _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(w1,w2),w3), c));
    _mm512_storeu_pd(a+j+16,_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(x1,x2),x3), c));
    _mm512_storeu_pd(a+j+24,_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(y1,y2),y3), c));
  }
  for (int64_t i = 2 + 32 * n4; i < n; i++) a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333;
}

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n == 0) return;
  if (n == 1) { a[0] = (b[0] + b[n-1] + b[n-2]) * 0.333; return; }
  a[0] = (b[0] + b[n-1] + b[n-2]) * 0.333;
  a[1] = (b[1] + b[0] + b[n-1]) * 0.333;
  if (n < 32768) { small_path(a, b, n); return; }
  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 2; i < n; i++) a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333;
}
