#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  const double threshold = 1.0;
  int64_t min_idx = INT64_MAX;

  #pragma omp parallel
  {
    const __m512d thresh = _mm512_set1_pd(threshold);
    int64_t local_min = INT64_MAX;
    const int64_t tid = omp_get_thread_num();
    const int64_t nt = omp_get_num_threads();
    const int64_t chunk = (LEN_1D + nt - 1) / nt;
    int64_t start = tid * chunk;
    int64_t end = start + chunk;
    if (end > LEN_1D) end = LEN_1D;

    int64_t i = start;
    for (; i < end && (i & 7) != 0; ++i) {
      if (a[i] > threshold) {
        local_min = i;
        break;
      }
    }
    if (local_min == INT64_MAX) {
      for (; i + 16 <= end; i += 16) {
        __m512d v0 = _mm512_loadu_pd(&a[i]);
        __m512d v1 = _mm512_loadu_pd(&a[i + 8]);
        __mmask8 m0 = _mm512_cmp_pd_mask(v0, thresh, _CMP_GT_OQ);
        __mmask8 m1 = _mm512_cmp_pd_mask(v1, thresh, _CMP_GT_OQ);
        if (m0 != 0) {
          local_min = i + __builtin_ctz(m0);
          break;
        }
        if (m1 != 0) {
          local_min = i + 8 + __builtin_ctz(m1);
          break;
        }
      }
      if (local_min == INT64_MAX) {
        for (; i + 8 <= end; i += 8) {
          __m512d v = _mm512_loadu_pd(&a[i]);
          __mmask8 mask = _mm512_cmp_pd_mask(v, thresh, _CMP_GT_OQ);
          if (mask != 0) {
            local_min = i + __builtin_ctz(mask);
            break;
          }
        }
        for (; i < end; ++i) {
          if (a[i] > threshold) {
            local_min = i;
            break;
          }
        }
      }
    }

    #pragma omp critical
    {
      if (local_min < min_idx) min_idx = local_min;
    }
  }

  if (min_idx == INT64_MAX) {
    out_index[0] = -1;
    out_value[0] = -1.0;
  } else {
    out_index[0] = min_idx;
    out_value[0] = a[min_idx];
  }
}
