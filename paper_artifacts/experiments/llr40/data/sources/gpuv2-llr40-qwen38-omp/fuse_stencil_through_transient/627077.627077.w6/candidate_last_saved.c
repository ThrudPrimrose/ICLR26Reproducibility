#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  int64_t n = LEN_1D;
  int len = (int)n;
  double *o1 = malloc((size_t)n * 8), *o2 = malloc((size_t)n * 8);
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) { o1[i] = 0; o2[i] = 0; }

  double *h = malloc((size_t)n * 8);
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) h[i] = (double)(i % 1000) * 0.125;

  /* 1) repeated map(to:) on same fresh buffer */
  double t0 = omp_get_wtime();
  #pragma omp target data map(to: h[0:len])
  { }
  double t1 = omp_get_wtime();
  #pragma omp target data map(to: h[0:len])
  { }
  double t2 = omp_get_wtime();
  #pragma omp target data map(to: h[0:len])
  { }
  double t3 = omp_get_wtime();
  printf("UP1 %.3f  UP2 %.3f  UP3 %.3f ms\n", (t1-t0)*1e3, (t2-t1)*1e3, (t3-t2)*1e3);

  /* is the upload real? read device image */
  double d0 = -999;
  #pragma omp target data map(to: h[0:len])
  {
    #pragma omp target map(from: d0)
    d0 = h[0];
  }
  printf("UP real? d0=%.3f host h0=%.3f\n", d0, h[0]);

  /* is download real? mark device, map(from:), check host */
  #pragma omp target data map(to: h[0:len])
  {
    #pragma omp target teams distribute parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) h[i] = 777.0;
  }
  #pragma omp target data map(from: h[0:len])
  { }
  printf("DL real? host h0=%.3f (777 if real)\n", h[0]);
  #pragma omp target data map(to: h[0:len])
  { }

  /* second fresh buffer: also slow first upload? */
  double *h2 = malloc((size_t)n * 8);
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) h2[i] = (double)(i % 77) * 0.25;
  double u0 = omp_get_wtime();
  #pragma omp target data map(to: h2[0:len])
  { }
  double u1 = omp_get_wtime();
  printf("UP-fresh-buffer2 %.3f ms\n", (u1-u0)*1e3);
  #pragma omp target exit data map(release: h2)
  free(h2);

  /* per-call design: cached a, 2 out buffers, update to per call */
  #pragma omp target enter data map(to: a[0:len])
  #pragma omp target enter data map(to: o1[0:len])
  #pragma omp target enter data map(to: o2[0:len])
  double *outs[2] = { o1, o2 };
  for (int k = 0; k < 6; ++k) {
    double *oc = outs[k % 2];
    double c0 = omp_get_wtime();
    #pragma omp target data map(tofrom: oc[0:len])
    {
      #pragma omp target teams distribute parallel for schedule(static)
      for (int64_t i = 1; i < n - 2; ++i)
        oc[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
    double c1 = omp_get_wtime();
    #pragma omp target update to(oc[0:len])
    double c2 = omp_get_wtime();
    printf("CALL%d region %.3f ms  writeback %.3f ms  total %.3f ms\n",
           k, (c1-c0)*1e3, (c2-c1)*1e3, (c2-c0)*1e3);
  }
  double r1 = (a[4] + a[5] + a[6]) * (a[5] + a[6] + a[7]);
  printf("CHECK o1[5]=%.9g ref=%.9g match=%d\n", o1[5], r1, o1[5] == r1);
  printf("CHECK o2[5]=%.9g match=%d\n", o2[5], o2[5] == r1);

  #pragma omp target exit data map(release: a, o1, o2)
  free(h); free(o1); free(o2);

  /* final correct answer for the harness */
  #pragma omp parallel for schedule(static)
  for (int64_t i = 1; i < n - 2; ++i)
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
}
