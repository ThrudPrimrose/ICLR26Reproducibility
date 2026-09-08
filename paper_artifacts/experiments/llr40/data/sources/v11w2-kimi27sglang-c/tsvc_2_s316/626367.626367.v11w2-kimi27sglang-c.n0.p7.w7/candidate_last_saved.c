#include <stdint.h>
#include <float.h>
#include <immintrin.h>
#include <omp.h>
#include <alloca.h>

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
  double global_min = a[0];

  if (LEN_1D <= 1) {
    result[0] = global_min;
    return;
  }

#if defined(__AVX512F__)
  if (LEN_1D <= 65536) {
    __m512d v = _mm512_set1_pd(global_min);
    int64_t i = 1;
    for (; i + 8 <= LEN_1D; i += 8) {
      v = _mm512_min_pd(v, _mm512_loadu_pd(&a[i]));
    }
    if (i < LEN_1D) {
      __mmask8 k = (__mmask8)((1u << (LEN_1D - i)) - 1u);
      __m512d tail = _mm512_mask_loadu_pd(_mm512_set1_pd(DBL_MAX), k, &a[i]);
      v = _mm512_min_pd(v, tail);
    }
    result[0] = _mm512_reduce_min_pd(v);
    return;
  }

  int max_threads = omp_get_max_threads();
  double *partial = (double *)alloca((size_t)max_threads * sizeof(double));

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    __m512d v = _mm512_set1_pd(DBL_MAX);
    #pragma omp for nowait schedule(static)
    for (int64_t i = 1; i < LEN_1D; i += 8) {
      __m512d chunk;
      if (i + 8 <= LEN_1D) {
        chunk = _mm512_loadu_pd(&a[i]);
      } else {
        __mmask8 k = (__mmask8)((1u << (LEN_1D - i)) - 1u);
        chunk = _mm512_mask_loadu_pd(_mm512_set1_pd(DBL_MAX), k, &a[i]);
      }
      v = _mm512_min_pd(v, chunk);
    }
    partial[tid] = _mm512_reduce_min_pd(v);
  }

  for (int t = 0; t < max_threads; ++t) {
    if (partial[t] < global_min) global_min = partial[t];
  }
#else
  #pragma omp parallel for reduction(min:global_min) schedule(static) if(LEN_1D > 65536)
  for (int64_t i = 1; i < LEN_1D; ++i) {
    if (a[i] < global_min) global_min = a[i];
  }
#endif

  result[0] = global_min;
}
