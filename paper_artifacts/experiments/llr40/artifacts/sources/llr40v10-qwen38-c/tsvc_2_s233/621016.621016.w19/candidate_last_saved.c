/* TSVC tsvc_2 s233 optimized.
 *
 * Part A: aa[j][i] = aa[j-1][i] + cc[j][i]  -> for each column i, a scan over j.
 *         Columns are independent -> parallelize over 8-column strips; the 8
 *         running sums stay in zmm registers; inner work is contiguous
 *         load/add/store.
 * Part B: bb[j][i] = bb[j][i-1] + cc[j][i]  -> for each row j, a scan over i.
 *         Rows are independent -> loop interchange; each row is a contiguous
 *         chunked prefix sum (2x8-wide SIMD, running broadcast accumulator).
 * Only the seeds aa[7][*] and bb[*][7] are read of the old outputs.
 */
#include <stdint.h>

#ifdef __AVX512F__
#include <immintrin.h>
#define HAS_512 1

static inline __m512d rsh8(__m512d v) {
  return _mm512_permutexvar_pd(_mm512_setr_epi32(15, 0, 1, 2, 3, 4, 5, 6), v);
}
static inline __m512d rsh16(__m512d v) {
  return _mm512_permutexvar_pd(_mm512_setr_epi32(15, 15, 0, 1, 2, 3, 4, 5), v);
}
static inline __m512d rsh32(__m512d v) {
  return _mm512_permutexvar_pd(_mm512_setr_epi32(15, 15, 15, 15, 0, 1, 2, 3), v);
}
static inline double lane7(__m512d v) {
  return _mm_hmov_sd(_mm512_extractf128_pd(v, 3));
}
#endif

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb,
                      const double *restrict cc, const int64_t LEN_2D)
{
  const int64_t N = LEN_2D;
  if (N <= 8) return;

  /* ---------------- Part A: column scans ---------------- */
  {
    const int64_t ncols = N - 8;
    const int64_t nfull = ncols / 8;          /* full 8-column strips        */
    const int64_t tail  = ncols - 8 * nfull;  /* 0..7 trailing columns       */

#ifdef HAS_512
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nfull; ++b) {
      const int64_t i0 = 8 + 8 * b;
      __m512d s0, s1, s2, s3, s4, s5, s6, s7;
      const double *seed = aa + 7 * N + i0;
      s0 = _mm512_loadu_pd(seed + 0);
      s1 = _mm512_loadu_pd(seed + 1);
      s2 = _mm512_loadu_pd(seed + 2);
      s3 = _mm512_loadu_pd(seed + 3);
      s4 = _mm512_loadu_pd(seed + 4);
      s5 = _mm512_loadu_pd(seed + 5);
      s6 = _mm512_loadu_pd(seed + 6);
      s7 = _mm512_loadu_pd(seed + 7);
      for (int64_t j = 8; j < N; ++j) {
        const double *crow = cc + j * N + i0;
        double *arow = aa + j * N + i0;
        __m512d v;
        v = _mm512_loadu_pd(crow + 0); s0 = _mm512_add_pd(s0, v); _mm512_storeu_pd(arow + 0, s0);
        v = _mm512_loadu_pd(crow + 1); s1 = _mm512_add_pd(s1, v); _mm512_storeu_pd(arow + 1, s1);
        v = _mm512_loadu_pd(crow + 2); s2 = _mm512_add_pd(s2, v); _mm512_storeu_pd(arow + 2, s2);
        v = _mm512_loadu_pd(crow + 3); s3 = _mm512_add_pd(s3, v); _mm512_storeu_pd(arow + 3, s3);
        v = _mm512_loadu_pd(crow + 4); s4 = _mm512_add_pd(s4, v); _mm512_storeu_pd(arow + 4, s4);
        v = _mm512_loadu_pd(crow + 5); s5 = _mm512_add_pd(s5, v); _mm512_storeu_pd(arow + 5, s5);
        v = _mm512_loadu_pd(crow + 6); s6 = _mm512_add_pd(s6, v); _mm512_storeu_pd(arow + 6, s6);
        v = _mm512_loadu_pd(crow + 7); s7 = _mm512_add_pd(s7, v); _mm512_storeu_pd(arow + 7, s7);
      }
    }
#endif
    /* tail columns (scalar; at most 7 strips of 1 col) */
#pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < tail; ++k) {
      const int64_t i = 8 + 8 * nfull + k;
      double s = aa[7 * N + i];
      for (int64_t j = 8; j < N; ++j) {
        s += cc[j * N + i];
        aa[j * N + i] = s;
      }
    }
  }

  /* ---------------- Part B: row scans (interchanged) ---------------- */
#pragma omp parallel for schedule(static)
  for (int64_t j = 8; j < N; ++j) {
    double *brow = bb + j * N;
    const double *crow = cc + j * N;
#ifdef HAS_512
    int64_t i = 8;
    const int64_t iend = N;
    double tsc = brow[7];
    __m512d tv = _mm512_set1_pd(tsc);
    for (; i + 16 <= iend; i += 16) {
      __m512d a = _mm512_loadu_pd(crow + i);
      __m512d d = _mm512_loadu_pd(crow + i + 8);
      a = _mm512_add_pd(a, tv);
      a = _mm512_add_pd(a, rsh8(a));
      a = _mm512_add_pd(a, rsh16(a));
      a = _mm512_add_pd(a, rsh32(a));
      tv = _mm512_set1_pd(lane7(a));
      d = _mm512_add_pd(d, tv);
      d = _mm512_add_pd(d, rsh8(d));
      d = _mm512_add_pd(d, rsh16(d));
      d = _mm512_add_pd(d, rsh32(d));
      _mm512_storeu_pd(brow + i, a);
      _mm512_storeu_pd(brow + i + 8, d);
      tv = _mm512_set1_pd(lane7(d));
    }
    tsc = _mm_cvtsd_f64(_mm512_extractf128_pd(tv, 0));
    for (; i < iend; ++i) { tsc += crow[i]; brow[i] = tsc; }
#else
    double t = brow[7];
    for (int64_t i = 8; i < N; ++i) { t += crow[i]; brow[i] = t; }
#endif
  }
}
