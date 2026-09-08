#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  double sum = 0.0;
  double maxa = 0.0, maxb = 0.0;
  for (int64_t i = 0; i < LEN_1D; ++i) {
    sum += a[i];
    b[i] = sum;
    double da = fabs(a[i]); if (da > maxa) maxa = da;
    double db = fabs(sum); if (db > maxb) maxb = db;
  }
  fprintf(stdout, "T=%d procs=%d n=%lld maxa=%.6e maxb=%.6e last=%.6e mid=%.6e a0=%.6e a1=%.6e\n",
          omp_get_max_threads(), omp_get_num_procs(), (long long)LEN_1D,
          maxa, maxb, b[LEN_1D-1], b[LEN_1D/2], a[0], a[1]);
  fflush(stdout);
}
