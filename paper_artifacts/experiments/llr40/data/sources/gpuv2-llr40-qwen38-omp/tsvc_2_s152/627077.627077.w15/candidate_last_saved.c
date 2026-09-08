#include <stdint.h>
#include <stdio.h>
static int ncall = 0;
void tsvc_2_s152_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  ncall++;
  if (ncall <= 20) {
    fprintf(stderr, "call %d: a=%p c=%p d=%p e=%p b=%p N=%lld\n",
            ncall, (void*)a,(void*)c,(void*)d,(void*)e,(void*)b,(long long)LEN_1D);
  }
  if (LEN_1D <= 0) return;
  #pragma omp target teams distribute \
      map(tofrom: a[0:LEN_1D]) map(from: b[0:LEN_1D]) \
      map(to: c[0:LEN_1D]) map(to: d[0:LEN_1D]) map(to: e[0:LEN_1D])
  for (int64_t i = 0; i < LEN_1D; ++i) {
    double t = d[i] * e[i]; b[i] = t; a[i] += t * c[i];
  }
}
