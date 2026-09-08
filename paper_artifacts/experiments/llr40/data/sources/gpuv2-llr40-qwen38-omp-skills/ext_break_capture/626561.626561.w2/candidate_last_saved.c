/* TSVC ext_break_capture: find first i with a[i] > K (K = 1.0), capture (i, a[i]).
 * GPU path: one map(to:) of the input, one fused kernel: per-element compare,
 * per-thread first-hit, cross-team/cross-thread integer min reduction (exact).
 * Host path for small inputs. */

#include <stdint.h>
#include <omp.h>

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  const double k = 1.0;

  if (LEN_1D <= 0) {
    out_index[0] = -1;
    out_value[0] = -1.0;
    return;
  }

  if (LEN_1D < (1 << 18)) {
    /* host scan: fully vectorizable early-exit loop */
    for (int64_t i = 0; i < LEN_1D; ++i) {
      if (a[i] > k) {
        out_index[0] = i;
        out_value[0] = a[i];
        return;
      }
    }
    out_index[0] = -1;
    out_value[0] = -1.0;
    return;
  }

  int64_t idx = INT64_MAX;
  int64_t th = INT64_MAX;

  #pragma omp target teams distribute parallel for reduction(min:idx) firstprivate(th) \
      map(to: a[0:LEN_1D])
  for (int64_t i = 0; i < LEN_1D; ++i) {
    if (th == INT64_MAX && a[i] > k) {
      th = i;
      idx = i; /* reduction variable: updated once per thread, min-merged at exit */
    }
  }

  if (idx == INT64_MAX) {
    out_index[0] = -1;
    out_value[0] = -1.0;
  } else {
    out_index[0] = idx;
    out_value[0] = a[idx];
  }
}
