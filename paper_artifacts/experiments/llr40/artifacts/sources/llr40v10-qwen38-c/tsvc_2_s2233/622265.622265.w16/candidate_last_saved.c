/* tsvc_2 s2233: two prefix scans.
 * Section 1: for each column i in [8,N): aa[j,i] = aa[7,i] + sum_{k=8..j} cc[k,i]  (scan over j)
 * Section 2: for each column j in [8,N): bb[i,j] = bb[7,j] + sum_{k=8..i} cc[k,j]  (scan over i)
 * The sequential carry lives in a register (or SIMD vector over the independent axis),
 * so each 16-lane chain is fully vectorizable and chains are independent -> threadable.
 */
#include <stdint.h>
#include <omp.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#define VEC 16
#elif defined(__AVX2__)
#include <immintrin.h>
#define VEC 4
#else
#define VEC 1
#endif

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D)
{
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  const int64_t n = N - 8;
  const int64_t V = VEC;
  const int64_t nvec = n / V;
  const int64_t vbase = 8 + nvec * V; /* first scalar lane */

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int nthr = omp_get_num_threads();

    /* Section 1: aa[j,i] += cumulative cc down column i; chain index = vector of i's. */
    for (int64_t v = tid; v < nvec; v += nthr) {
      const int64_t i0 = 8 + v * V;
#if VEC == 16
      __m512d carry = _mm512_loadu_pd(aa + 7 * N + i0);
      for (int64_t j = 8; j < N; ++j) {
        carry = _mm512_add_pd(carry, _mm512_loadu_pd(cc + j * N + i0));
        _mm512_storeu_pd(aa + j * N + i0, carry);
      }
#elif VEC == 4
      __m256d carry = _mm256_loadu_pd(aa + 7 * N + i0);
      for (int64_t j = 8; j < N; ++j) {
        carry = _mm256_add_pd(carry, _mm256_loadu_pd(cc + j * N + i0));
        _mm256_storeu_pd(aa + j * N + i0, carry);
      }
#else
      double carry = aa[7 * N + i0];
      for (int64_t j = 8; j < N; ++j) {
        carry += cc[j * N + i0];
        aa[j * N + i0] = carry;
      }
#endif
    }
    for (int64_t i = vbase + tid; i < N; i += nthr) {
      double carry = aa[7 * N + i];
      for (int64_t j = 8; j < N; ++j) {
        carry += cc[j * N + i];
        aa[j * N + i] = carry;
      }
    }

    /* Section 2: bb[i,j] = cumulative cc over i for row position j; chain index = vector of j's. */
    for (int64_t v = tid; v < nvec; v += nthr) {
      const int64_t j0 = 8 + v * V;
#if VEC == 16
      __m512d carry = _mm512_loadu_pd(bb + 7 * N + j0);
      for (int64_t i = 8; i < N; ++i) {
        carry = _mm512_add_pd(carry, _mm512_loadu_pd(cc + i * N + j0));
        _mm512_storeu_pd(bb + i * N + j0, carry);
      }
#elif VEC == 4
      __m256d carry = _mm256_loadu_pd(bb + 7 * N + j0);
      for (int64_t i = 8; i < N; ++i) {
        carry = _mm256_add_pd(carry, _mm256_loadu_pd(cc + i * N + j0));
        _mm256_storeu_pd(bb + i * N + j0, carry);
      }
#else
      double carry = bb[7 * N + j0];
      for (int64_t i = 8; i < N; ++i) {
        carry += cc[i * N + j0];
        bb[i * N + j0] = carry;
      }
#endif
    }
    for (int64_t j = vbase + tid; j < N; j += nthr) {
      double carry = bb[7 * N + j];
      for (int64_t i = 8; i < N; ++i) {
        carry += cc[i * N + j];
        bb[i * N + j] = carry;
      }
    }
  }
}
