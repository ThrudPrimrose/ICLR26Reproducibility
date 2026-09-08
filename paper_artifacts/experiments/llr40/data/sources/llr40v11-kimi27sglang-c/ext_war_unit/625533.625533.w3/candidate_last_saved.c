#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D <= 1) return;
  const int64_t n = LEN_1D - 1;

  if (n < 4096) {
    for (int64_t i = 0; i < n; ++i) {
      a[i] = a[i + 1] + b[i];
    }
    return;
  }

  const int max_threads = omp_get_max_threads();
  int64_t *restrict starts = (int64_t *)__builtin_alloca((max_threads + 1) * sizeof(int64_t));
  double *restrict boundary = (double *)__builtin_alloca(max_threads * sizeof(double));

  #pragma omp parallel
  {
    const int t = omp_get_thread_num();
    const int nt = omp_get_num_threads();
    starts[t] = (int64_t)t * n / nt;
    starts[nt] = n;
    #pragma omp barrier
    // Capture the single boundary element this chunk reads from the next chunk.
    boundary[t] = a[starts[t + 1]];
    #pragma omp barrier
    const int64_t lo = starts[t];
    const int64_t hi = starts[t + 1];
    int64_t i = lo;
#if defined(__AVX2__)
    const int64_t vec_end = hi - 3;
    for (; i < vec_end; i += 4) {
      __m256d av = _mm256_loadu_pd(a + i + 1);
      __m256d bv = _mm256_loadu_pd(b + i);
      __m256d sv = _mm256_add_pd(av, bv);
      _mm256_storeu_pd(a + i, sv);
    }
#endif
    for (; i < hi; ++i) {
      double av = (i + 1 == hi) ? boundary[t] : a[i + 1];
      a[i] = av + b[i];
    }
  }
}
