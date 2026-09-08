#include <stdint.h>
#include <stdlib.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n < 2) return;
  const int64_t T = 8;           /* tile size: serial chain within a tile */
  const int64_t m = n - 1;       /* number of output elements a[0..n-2] */
  const int64_t ntiles = (m + T - 1) / T;
  double *border = (double *)malloc(ntiles * sizeof(double));
  #pragma omp target data map(tofrom: a[0:n]) map(to: b[0:n]) map(alloc: border[0:ntiles])
  {
    /* Phase 0: save each tile's boundary element a[hi_k] (original), which the
       next tile will overwrite. Tiny pass (ntiles elements). */
    #pragma omp target
    #pragma omp teams distribute parallel for
    for (int64_t k = 0; k < ntiles; ++k) {
      int64_t hi = (k + 1) * T; if (hi > m) hi = m;
      border[k] = a[hi];
    }
    /* Phase 1: per-tile shift+add. Within a tile the unit WAR is resolved by
       ascending order; the last element of each tile uses the saved border. */
    #pragma omp target
    #pragma omp teams distribute parallel for
    for (int64_t k = 0; k < ntiles; ++k) {
      int64_t lo = k * T;
      int64_t hi = (k + 1) * T; if (hi > m) hi = m;
      for (int64_t i = lo; i + 1 < hi; ++i) a[i] = a[i + 1] + b[i];
      if (hi > lo) a[hi - 1] = border[k] + b[hi - 1];
    }
  }
  free(border);
}
