#include <stdint.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
  const int64_t L = LEN_2D;
  printf("L=%lld\n", (long long)L);
  {
    const char* names[5] = {0};
    (void)names;
    const double* arrs[5] = {a, aa, b, bb, c};
    const char* nm[5] = {"a","aa","b","bb","c"};
    for (int k = 0; k < 5; k++) {
      int64_t n = (k < 3) ? L : L*L;
      double mn = 1e300, mx = -1e300, amx = 0.0, sum = 0.0;
      int64_t i;
      #pragma omp parallel for reduction(min:mn) reduction(max:mx,amx,sum)
      for (i = 0; i < n; i++) {
        double v = arrs[k][i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        double av = fabs(v);
        if (av > amx) amx = av;
        sum += v;
      }
      printf("%s: min=%.17g max=%.17g absmax=%.17g sum=%.17g\n", nm[k], mn, mx, amx, sum);
    }
    // first few elements
    printf("a[:4]= %.17g %.17g %.17g %.17g\n", a[0], a[1], a[2], a[3]);
    printf("b[:4]= %.17g %.17g %.17g %.17g\n", b[0], b[1], b[2], b[3]);
    printf("c[:4]= %.17g %.17g %.17g %.17g\n", c[0], c[1], c[2], c[3]);
    printf("aa[0][:4]= %.17g %.17g %.17g %.17g\n", aa[0], aa[1], aa[2], aa[3]);
    printf("bb[0][:4]= %.17g %.17g %.17g %.17g\n", bb[0], bb[1], bb[2], bb[3]);
    printf("aa[%lld-1]= %.17g aa[%lld-1,1]= %.17g\n", (long long)(L*L-1), aa[L*L-1], (long long)(L*L-1), aa[L*L-1+1>0? (L-1)*L+1 : 0]);
  }
  volatile double sink = a[0] + aa[L*L-1] + b[0] + bb[L*L-1] + c[0];
  if (sink == 12345.6789) printf("unexpected exact value\n");
  fflush(stdout);
}
