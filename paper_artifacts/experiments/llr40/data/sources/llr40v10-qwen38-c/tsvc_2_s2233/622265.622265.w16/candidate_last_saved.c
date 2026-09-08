/* tsvc_2 s2233 optimized.
 *
 * Semantics: for i in [8,N): for j in [8,N): aa[j,i] = aa[j-1,i] + cc[j,i]   (column scan, carry in aa[7,i])
 *            and                  bb[i,j] = bb[i-1,j] + cc[i,j]             (row scan,    carry in bb[7,j])
 * Both are prefix sums with a per-column/per-row carry that stays in a register, so:
 *  - vectorize over the independent axis (columns for aa, rows for bb),
 *  - parallelize over the chains (they are independent of each other),
 *  - interleave 4 chains per step for instruction-level parallelism.
 *
 * The SIMD level is picked by a tiny runtime self-test (512 -> 256 -> scalar): some
 * virtualized CPUs advertise AVX-512F but truncate 512-bit memory ops to 256 bits,
 * which corrupts results; the self-test catches exactly that.
 */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

static int simd_level(void)
{
#if defined(__AVX512F__)
  {
    double a[16], b[16];
    int i, bad;
    for (i = 0; i < 16; ++i) { a[i] = (double)(i + 1); b[i] = -1.0; }
    {
      __m512d v = _mm512_loadu_pd(a);
      v = _mm512_add_pd(v, _mm512_loadu_pd(a));
      _mm512_storeu_pd(b, v);
    }
    for (i = 0, bad = 0; i < 16; ++i)
      if (b[i] != 2.0 * (double)(i + 1)) bad = 1;
    if (!bad) return 2; /* 512-bit OK */
  }
#endif
#if defined(__AVX2__)
  {
    double a2[8], b2[8];
    int i, bad;
    for (i = 0; i < 8; ++i) { a2[i] = (double)(i + 1); b2[i] = -1.0; }
    {
      __m256d v = _mm256_loadu_pd(a2);
      v = _mm256_add_pd(v, _mm256_loadu_pd(a2));
      _mm256_storeu_pd(b2, v);
    }
    for (i = 0, bad = 0; i < 8; ++i)
      if (b2[i] != 2.0 * (double)(i + 1)) bad = 1;
    if (!bad) return 1; /* 256-bit OK */
  }
#endif
  return 0;
}

/* A "section" = prefix scan along the outer axis with carry in row/col 7.
 * For aa: chains over columns i, scan over j (contiguous 4/8 doubles within a chain).
 * For bb: chains over rows-j, scan over i (same shape, different array).
 * The body is identical:  OUT[scan*STRIDE + off] += IN[scan*STRIDE + off] cumulatively.
 */

#define DEFINE_SCAN(NAME, KIND)                                                              \
  static void NAME(double *restrict OUT, const double *restrict IN, const int64_t N)         \
  {                                                                                          \
    const int lvl = simd_level();                                                            \
    _Pragma("omp parallel")                                                                  \
    {                                                                                        \
      const int tid = omp_get_thread_num();                                                  \
      const int nthr = omp_get_num_threads();                                                \
      if (lvl == 2 && N > 8) {                                                               \
        const int64_t n32 = (N - 8) / 32;                                                    \
        for (int64_t p = tid; p < n32; p += nthr) {                                          \
          const int64_t o0 = 8 + 32 * p;                                                     \
          __m512d s0 = _mm512_loadu_pd(OUT + 7 * N + o0);                                    \
          __m512d s1 = _mm512_loadu_pd(OUT + 7 * N + o0 + 16);                               \
          for (int64_t k = 8; k < N; ++k) {                                                  \
            s0 = _mm512_add_pd(s0, _mm512_loadu_pd(IN + k * N + o0));                        \
            s1 = _mm512_add_pd(s1, _mm512_loadu_pd(IN + k * N + o0 + 16));                   \
            _mm512_storeu_pd(OUT + k * N + o0, s0);                                          \
            _mm512_storeu_pd(OUT + k * N + o0 + 16, s1);                                     \
          }                                                                                  \
        }                                                                                    \
        for (int64_t r = 8 + 32 * n32 + tid; r < N; r += nthr) {                             \
          double s = OUT[7 * N + r];                                                         \
          for (int64_t k = 8; k < N; ++k) { s += IN[k * N + r]; OUT[k * N + r] = s; }        \
        }                                                                                    \
      } else if (lvl == 1 && N > 8) {                                                        \
        const int64_t n16 = (N - 8) / 16;                                                    \
        for (int64_t g = tid; g < n16; g += nthr) {                                          \
          const int64_t o0 = 8 + 16 * g;                                                     \
          __m256d s0 = _mm256_loadu_pd(OUT + 7 * N + o0);                                    \
          __m256d s1 = _mm256_loadu_pd(OUT + 7 * N + o0 + 4);                                \
          __m256d s2 = _mm256_loadu_pd(OUT + 7 * N + o0 + 8);                                \
          __m256d s3 = _mm256_loadu_pd(OUT + 7 * N + o0 + 12);                               \
          for (int64_t k = 8; k < N; ++k) {                                                  \
            s0 = _mm256_add_pd(s0, _mm256_loadu_pd(IN + k * N + o0));                        \
            s1 = _mm256_add_pd(s1, _mm256_loadu_pd(IN + k * N + o0 + 4));                    \
            s2 = _mm256_add_pd(s2, _mm256_loadu_pd(IN + k * N + o0 + 8));                    \
            s3 = _mm256_add_pd(s3, _mm256_loadu_pd(IN + k * N + o0 + 12));                   \
            _mm256_storeu_pd(OUT + k * N + o0, s0);                                          \
            _mm256_storeu_pd(OUT + k * N + o0 + 4, s1);                                      \
            _mm256_storeu_pd(OUT + k * N + o0 + 8, s2);                                      \
            _mm256_storeu_pd(OUT + k * N + o0 + 12, s3);                                     \
          }                                                                                  \
        }                                                                                    \
        for (int64_t r = 8 + 16 * n16 + tid; r < N; r += nthr) {                             \
          double s = OUT[7 * N + r];                                                         \
          for (int64_t k = 8; k < N; ++k) { s += IN[k * N + r]; OUT[k * N + r] = s; }        \
        }                                                                                    \
      } else {                                                                               \
        for (int64_t r = 8 + tid; r < N; r += nthr) {                                        \
          double s = OUT[7 * N + r];                                                         \
          for (int64_t k = 8; k < N; ++k) { s += IN[k * N + r]; OUT[k * N + r] = s; }        \
        }                                                                                    \
      }                                                                                      \
    }                                                                                        \
  }

DEFINE_SCAN(sec_aa, AA)
DEFINE_SCAN(sec_bb, BB)

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D)
{
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  sec_aa(aa, cc, N);
  sec_bb(bb, cc, N);
}
