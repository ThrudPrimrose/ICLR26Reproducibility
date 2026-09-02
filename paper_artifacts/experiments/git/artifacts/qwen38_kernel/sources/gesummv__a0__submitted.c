/* gesummv: out = alpha * A*x + beta * B*x  (A,B: N x N row-major fp64)
 *
 * Memory-bound streaming GEMV. Two dot products per row (A and B), fused in one
 * pass so x is fetched once per row. AVX-512 (8-wide) / AVX2 (4-wide) inner
 * loop with 4 accumulator lanes; OpenMP splits rows across cores.
 */
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

/* ---------------- AVX-512 row kernel ---------------- */
#if defined(__AVX512F__) && defined(__FMA__)

static inline void row_avx512(const double *restrict a, const double *restrict b,
                              const double *restrict x, int64_t N,
                              double alpha, double beta, double *restrict o)
{
    __m512d da0 = _mm512_setzero_pd(), da1 = _mm512_setzero_pd(),
            da2 = _mm512_setzero_pd(), da3 = _mm512_setzero_pd();
    __m512d db0 = _mm512_setzero_pd(), db1 = _mm512_setzero_pd(),
            db2 = _mm512_setzero_pd(), db3 = _mm512_setzero_pd();
    int64_t j = 0;
    int64_t lim = N - (N & 31);
    for (; j < lim; j += 32) {
        const double *pa = a + j, *pb = b + j, *px = x + j;
        da0 = _mm512_fmadd_pd(_mm512_loadu_pd(pa),       _mm512_loadu_pd(px),       da0);
        da1 = _mm512_fmadd_pd(_mm512_loadu_pd(pa + 8),   _mm512_loadu_pd(px + 8),   da1);
        da2 = _mm512_fmadd_pd(_mm512_loadu_pd(pa + 16),  _mm512_loadu_pd(px + 16),  da2);
        da3 = _mm512_fmadd_pd(_mm512_loadu_pd(pa + 24),  _mm512_loadu_pd(px + 24),  da3);
        db0 = _mm512_fmadd_pd(_mm512_loadu_pd(pb),       _mm512_loadu_pd(px),       db0);
        db1 = _mm512_fmadd_pd(_mm512_loadu_pd(pb + 8),   _mm512_loadu_pd(px + 8),   db1);
        db2 = _mm512_fmadd_pd(_mm512_loadu_pd(pb + 16),  _mm512_loadu_pd(px + 16),  db2);
        db3 = _mm512_fmadd_pd(_mm512_loadu_pd(pb + 24),  _mm512_loadu_pd(px + 24),  db3);
    }
    __m512d da = _mm512_add_pd(_mm512_add_pd(da0, da1), _mm512_add_pd(da2, da3));
    __m512d db = _mm512_add_pd(_mm512_add_pd(db0, db1), _mm512_add_pd(db2, db3));
    double sda = _mm512_reduce_add_pd(da);
    double sdb = _mm512_reduce_add_pd(db);
    for (; j < N; j++) { sda += a[j] * x[j]; sdb += b[j] * x[j]; }
    *o = alpha * sda + beta * sdb;
}

static void rows_avx512(const double *restrict A, const double *restrict B,
                        double *restrict out, const double *restrict x,
                        int64_t N, double alpha, double beta)
{
    #pragma omp parallel for schedule(static, 8)
    for (int64_t i = 0; i < N; i++)
        row_avx512(A + i * N, B + i * N, x, N, alpha, beta, out + i);
}

#endif /* AVX512 */

/* ---------------- AVX2 row kernel ---------------- */
#if !defined(__AVX512F__) && defined(__AVX2__) && defined(__FMA__)

static inline void row_avx2(const double *restrict a, const double *restrict b,
                            const double *restrict x, int64_t N,
                            double alpha, double beta, double *restrict o)
{
    __m256d da0 = _mm256_setzero_pd(), da1 = _mm256_setzero_pd(),
            da2 = _mm256_setzero_pd(), da3 = _mm256_setzero_pd();
    __m256d db0 = _mm256_setzero_pd(), db1 = _mm256_setzero_pd(),
            db2 = _mm256_setzero_pd(), db3 = _mm256_setzero_pd();
    int64_t j = 0;
    int64_t lim = N - (N & 15);
    for (; j < lim; j += 16) {
        const double *pa = a + j, *pb = b + j, *px = x + j;
        da0 = _mm256_fmadd_pd(_mm256_loadu_pd(pa),      _mm256_loadu_pd(px),      da0);
        da1 = _mm256_fmadd_pd(_mm256_loadu_pd(pa + 4),  _mm256_loadu_pd(px + 4),  da1);
        da2 = _mm256_fmadd_pd(_mm256_loadu_pd(pa + 8),  _mm256_loadu_pd(px + 8),  da2);
        da3 = _mm256_fmadd_pd(_mm256_loadu_pd(pa + 12), _mm256_loadu_pd(px + 12), da3);
        db0 = _mm256_fmadd_pd(_mm256_loadu_pd(pb),      _mm256_loadu_pd(px),      db0);
        db1 = _mm256_fmadd_pd(_mm256_loadu_pd(pb + 4),  _mm256_loadu_pd(px + 4),  db1);
        db2 = _mm256_fmadd_pd(_mm256_loadu_pd(pb + 8),  _mm256_loadu_pd(px + 8),  db2);
        db3 = _mm256_fmadd_pd(_mm256_loadu_pd(pb + 12), _mm256_loadu_pd(px + 12), db3);
    }
    __m256d da = _mm256_add_pd(_mm256_add_pd(da0, da1), _mm256_add_pd(da2, da3));
    __m256d db = _mm256_add_pd(_mm256_add_pd(db0, db1), _mm256_add_pd(db2, db3));
    double sda = _mm256_reduce_add_pd(da);
    double sdb = _mm256_reduce_add_pd(db);
    for (; j < N; j++) { sda += a[j] * x[j]; sdb += b[j] * x[j]; }
    *o = alpha * sda + beta * sdb;
}

static void rows_avx2(const double *restrict A, const double *restrict B,
                      double *restrict out, const double *restrict x,
                      int64_t N, double alpha, double beta)
{
    #pragma omp parallel for schedule(static, 8)
    for (int64_t i = 0; i < N; i++)
        row_avx2(A + i * N, B + i * N, x, N, alpha, beta, out + i);
}

#endif /* AVX2 only */

/* ---------------- scalar fallback ---------------- */
static void rows_scalar(const double *restrict A, const double *restrict B,
                        double *restrict out, const double *restrict x,
                        int64_t N, double alpha, double beta)
{
    #pragma omp parallel for schedule(static, 8)
    for (int64_t i = 0; i < N; i++) {
        double sda = 0.0, sdb = 0.0;
        const double *a = A + i * N, *b = B + i * N;
        for (int64_t j = 0; j < N; j++) {
            sda += a[j] * x[j];
            sdb += b[j] * x[j];
        }
        out[i] = alpha * sda + beta * sdb;
    }
}

/* ---------------- entry point ---------------- */
void gesummv_fp64(const double *restrict A, const double *restrict B,
                  double *restrict out, const double *restrict x,
                  int64_t N, double alpha, double beta)
{
    if (N <= 0) return;
    (void)rows_scalar; /* keep fallback available on scalar-only builds */

    /* below this, OpenMP fork/join overhead would eat the win */
    if (N < 384) {
#if defined(__AVX512F__) && defined(__FMA__)
        for (int64_t i = 0; i < N; i++)
            row_avx512(A + i * N, B + i * N, x, N, alpha, beta, out + i);
#elif defined(__AVX2__) && defined(__FMA__)
        for (int64_t i = 0; i < N; i++)
            row_avx2(A + i * N, B + i * N, x, N, alpha, beta, out + i);
#else
        rows_scalar(A, B, out, x, N, alpha, beta);
#endif
        return;
    }

#if defined(__AVX512F__) && defined(__FMA__)
    rows_avx512(A, B, out, x, N, alpha, beta);
#elif defined(__AVX2__) && defined(__FMA__)
    rows_avx2(A, B, out, x, N, alpha, beta);
#else
    rows_scalar(A, B, out, x, N, alpha, beta);
#endif
}
