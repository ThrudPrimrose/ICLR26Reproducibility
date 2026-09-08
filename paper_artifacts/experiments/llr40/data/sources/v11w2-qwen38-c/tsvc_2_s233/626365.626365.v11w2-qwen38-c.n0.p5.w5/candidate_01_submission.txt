#include <stdint.h>
#include <omp.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t N) {
  if (N <= 8) return;

  #pragma omp parallel
  {
    /* Part 1: aa[j][i] = aa[j-1][i] + cc[j][i]; per-column scan over rows j.
       Columns are independent. Process 8 contiguous columns per thread block:
       contiguous 64B memory per step, 8 independent accumulation chains. */
    #pragma omp for schedule(static) nowait
    for (int64_t i0 = 8; i0 < N; i0 += 8) {
      int64_t i1 = i0 + 8; if (i1 > N) i1 = N;
#if defined(__AVX512F__)
      if (i1 - i0 == 8) {
        /* 8 contiguous columns per step; unaligned intrinsics (N need not be 64B-multiple) */
        __m512d v = _mm512_loadu_pd(aa + 7 * N + i0);
        const double *ccp = cc + 8 * N + i0;
        double *aap = aa + 8 * N + i0;
        for (int64_t j = 8; j < N; ++j) {
          v = _mm512_add_pd(v, _mm512_loadu_pd(ccp));
          _mm512_storeu_pd(aap, v);
          ccp += N;
          aap += N;
        }
        continue;
      }
#endif
      for (int64_t i = i0; i < i1; ++i) {
        double acc = aa[7 * N + i];
        for (int64_t j = 8; j < N; ++j) {
          acc += cc[j * N + i];
          aa[j * N + i] = acc;
        }
      }
    }

    /* Part 2: bb[j][i] = bb[j][i-1] + cc[j][i]; per-row scan over cols i.
       Rows are independent. Process 8 rows per thread block: contiguous
       row streams, 8 independent accumulation chains. */
    #pragma omp for schedule(static) nowait
    for (int64_t r0 = 8; r0 < N; r0 += 8) {
      int64_t r1 = r0 + 8; if (r1 > N) r1 = N;
      int64_t base = r0 * N;
      int64_t rp[8];
      double a[8];
      for (int64_t k = 0; k < r1 - r0; ++k) {
        rp[k] = base + k * N;
        a[k] = bb[rp[k] + 7];
      }
      for (int64_t i = 8; i < N; ++i) {
        for (int64_t k = 0; k < r1 - r0; ++k) {
          a[k] += cc[rp[k] + i];
          bb[rp[k] + i] = a[k];
        }
      }
    }
  }
}
