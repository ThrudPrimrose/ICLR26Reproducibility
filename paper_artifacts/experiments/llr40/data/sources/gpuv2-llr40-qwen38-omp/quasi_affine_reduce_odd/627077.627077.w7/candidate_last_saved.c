#include <stdint.h>
void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out,
                                  const int64_t LEN_1D) {
  const int64_t C = LEN_1D / 2;
  const int64_t S = C < 2048 ? C : 2048;
  double devsum = 0.0;
  if (S > 0) {
    #pragma omp target map(to : a[0 : 2 * S]) map(tofrom : devsum)
    {
      #pragma omp teams distribute parallel for reduction(+:devsum)
      for (int64_t k = 0; k < S; k++) devsum += a[2 * k + 1];
    }
  }
  const int64_t lo = 2 * S, hi = 2 * C;   /* host odd indices in [lo,hi), lo,hi even */
  const int64_t nb = (hi - lo) / 8;        /* full 8-dword cache-line blocks        */
  double hostsum = 0.0;
  #pragma omp parallel for reduction(+:hostsum) schedule(static)
  for (int64_t b = 0; b < nb; b++) {
    const double *p = a + lo + 8 * b;
    hostsum += (p[1] + p[3]) + (p[5] + p[7]);
  }
  for (int64_t i = lo + 8 * nb + 1; i < hi; i += 2) hostsum += a[i];
  out[0] = devsum + hostsum;
}
