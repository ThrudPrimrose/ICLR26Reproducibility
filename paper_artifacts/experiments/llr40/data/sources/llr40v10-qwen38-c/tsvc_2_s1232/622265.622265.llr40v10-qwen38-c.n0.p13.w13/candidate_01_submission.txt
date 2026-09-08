/* TSVC tsvc_2 s1232: for j in [0,N): for i in [j*VLEN, N): aa[i*N+j] = bb[i*N+j] + cc[i*N+j]
 * (row-major, element (i,j) at i*N+j; N = LEN_2D, V = VLEN).
 *
 * Transposed view: row i holds columns j = 0..min(i/V, N-1)  (condition j*V <= i).
 * Each row segment is CONTIGUOUS in memory: aa[i*N .. i*N + jmax].
 * Stream rows with 512-bit SIMD; parallelize over rows (OpenMP).
 */
#include <stdint.h>
#include <immintrin.h>

static void segment_add(double *restrict a, const double *restrict b, const double *restrict c, int64_t len) {
  int64_t k = 0;
#if defined(__AVX512F__)
  for (; k + 8 <= len; k += 8)
    _mm512_storeu_pd(a + k, _mm512_add_pd(_mm512_loadu_pd(b + k), _mm512_loadu_pd(c + k)));
#elif defined(__AVX2__)
  for (; k + 4 <= len; k += 4)
    _mm256_storeu_pd(a + k, _mm256_add_pd(_mm256_loadu_pd(b + k), _mm256_loadu_pd(c + k)));
#endif
  for (; k < len; ++k) a[k] = b[k] + c[k];
}

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       const int64_t LEN_2D, const int64_t VLEN) {
  const int64_t N = LEN_2D;
  if (N <= 0) return;
  const int posV = (VLEN > 0);

  if (!posV || N * N < (1ll << 16)) {  /* V<=0: full matrix; or tiny problem: serial */
    for (int64_t i = 0; i < N; ++i) {
      int64_t jmax = posV ? (i / VLEN) : (N - 1);
      if (jmax >= N) jmax = N - 1;
      segment_add(aa + i * N, bb + i * N, cc + i * N, jmax + 1);
    }
    return;
  }

#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t i = 0; i < N; ++i) {
    int64_t jmax = i / VLEN;
    if (jmax >= N) jmax = N - 1;
    segment_add(aa + i * N, bb + i * N, cc + i * N, jmax + 1);
  }
}
