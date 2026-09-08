#include <stdint.h>

/* TSVC s235:
 *   for i: a[i] += b[i]*c[i];  aa[j,i] = aa[j-1,i] + bb[j,i]*a[i]  (j=1..N-1)
 * Rows of aa form a serial scan in j; each row is independent over i.
 */
void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b,
                      const double *restrict bb, const double *restrict c, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  for (int64_t i = 0; i < N; ++i)
    a[i] += b[i] * c[i];
  for (int64_t j = 1; j < N; ++j) {
    const double *restrict prev = aa + (j - 1) * N;
    const double *restrict bbj = bb + j * N;
    double *restrict cur = aa + j * N;
    for (int64_t i = 0; i < N; ++i)
      cur[i] = prev[i] + bbj[i] * a[i];
  }
}
