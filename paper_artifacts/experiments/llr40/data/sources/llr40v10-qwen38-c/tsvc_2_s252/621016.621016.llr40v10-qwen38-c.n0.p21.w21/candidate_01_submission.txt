#include <stdint.h>
#include <omp.h>

__attribute__((optimize("fp-contract=off")))
static void body(double *restrict a, const double *restrict b, const double *restrict c, const int64_t lo, const int64_t hi) {
  for (int64_t i = lo; i < hi; ++i) a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
}

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  a[0] = b[0] * c[0];
  if (LEN_1D == 1) return;
  #pragma omp parallel
  {
    const int64_t nt  = omp_get_num_threads();
    const int64_t tid = omp_get_thread_num();
    const int64_t span = LEN_1D - 1;
    const int64_t per  = span / nt;
    const int64_t rem  = span % nt;
    const int64_t len  = per + (tid < rem ? 1 : 0);
    const int64_t lo = 1 + tid * per + (tid < rem ? tid : rem);
    body(a, b, c, lo, lo + len);
  }
}
