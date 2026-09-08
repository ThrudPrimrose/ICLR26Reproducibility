#include <stdint.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  const int64_t M = LEN_1D - 1;   /* iterations: i = 0 .. M-1 */
  if (M <= 0) return;

  if (M < (1 << 15)) {
    /* small trip counts: exact serial, bit-identical to the reference */
    for (int64_t i = 0; i < M; ++i) a[i] = a[i + 1] + b[i];
    return;
  }

  const double an1 = a[LEN_1D - 1];

  #pragma omp target data map(to: b[0:LEN_1D]) map(tofrom: a[0:LEN_1D])
  {
    double S = 0.0;
    #pragma omp target teams distribute parallel for simd reduction(+:S)
    for (int64_t i = 0; i < M; ++i) S += b[i];

    double s = 0.0;
    #pragma omp target teams distribute parallel for reduction(inscan,+:s)
    for (int64_t i = 0; i < M; ++i) {
      s += b[i];
      #pragma omp scan inclusive(s)
      a[i] = an1 + S - s + b[i];
    }
  }
}
