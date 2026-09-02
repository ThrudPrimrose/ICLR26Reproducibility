#include <stdint.h>
#include <omp.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
  int64_t n4 = (LEN_1D / 4) * 4;
  #pragma omp parallel for simd reduction(+:s0,s1,s2,s3,t0,t1,t2,t3)
  for (int64_t i = 0; i < n4; i += 4) {
    double c0 = c[i];
    double c1 = c[i + 1];
    double c2 = c[i + 2];
    double c3 = c[i + 3];
    double a0 = c0 + d[i];
    double a1 = c1 + d[i + 1];
    double a2 = c2 + d[i + 2];
    double a3 = c3 + d[i + 3];
    double b0 = c0 + e[i];
    double b1 = c1 + e[i + 1];
    double b2 = c2 + e[i + 2];
    double b3 = c3 + e[i + 3];
    a[i]     = a0;
    a[i + 1] = a1;
    a[i + 2] = a2;
    a[i + 3] = a3;
    b[i]     = b0;
    b[i + 1] = b1;
    b[i + 2] = b2;
    b[i + 3] = b3;
    s0 += a0; s1 += a1; s2 += a2; s3 += a3;
    t0 += b0; t1 += b1; t2 += b2; t3 += b3;
  }
  for (int64_t i = n4; i < LEN_1D; ++i) {
    double ci = c[i];
    double ai = ci + d[i];
    double bi = ci + e[i];
    a[i] = ai;
    b[i] = bi;
    s0 += ai;
    t0 += bi;
  }
  b[0] = ((s0 + s1) + (s2 + s3)) + ((t0 + t1) + (t2 + t3));
}
