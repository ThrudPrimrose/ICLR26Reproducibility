#include <stdint.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c,
                       double *restrict d, const int64_t LEN_1D) {
  const int64_t n = LEN_1D - 1;
  if (n <= 0) return;
  const int64_t m = n - 1;

  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < m; i++) {
    double bi = b[i];
    double ci = c[i];
    double bn = b[i + 1];
    double cn = c[i + 1];
    double ai = bi * bi + ci * ci + bi + ci;
    double an = bn * bn + cn * cn + bn + cn;
    a[i] = ai;
    d[i] = ai + an;
  }

  {
    double bi = b[m];
    double ci = c[m];
    double ai = bi * bi + ci * ci + bi + ci;
    a[m] = ai;
    d[m] = ai + a[n];
  }
}
