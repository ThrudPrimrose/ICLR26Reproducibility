#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

static void run_body(double *restrict a, double *restrict aa, const double *restrict b,
                     const double *restrict bb, const double *restrict c, const int64_t LEN_2D) {
  #pragma omp target map(tofrom: a[0:LEN_2D]) map(to: b[0:LEN_2D], c[0:LEN_2D]) \
                         map(tofrom: aa[0:LEN_2D * LEN_2D]) map(to: bb[0:LEN_2D * LEN_2D])
  {
    #pragma omp teams distribute
    for (int64_t i = 0; i < LEN_2D; ++i) {
      a[i] += b[i] * c[i];
      double prev = aa[i];
      for (int64_t j = 1; j < LEN_2D; ++j) {
        prev = bb[j * LEN_2D + i] * a[i] + prev;
        aa[j * LEN_2D + i] = prev;
      }
    }
  }
}

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
  static int callno = 0;
  callno++;
  if (callno == 1) {
    const int64_t N2 = LEN_2D * LEN_2D;
    double t0 = omp_get_wtime();
    #pragma omp target enter data map(to: a[0:LEN_2D]) map(to: b[0:LEN_2D], c[0:LEN_2D]) \
                                       map(to: aa[0:N2]) map(to: bb[0:N2])
    double t1 = omp_get_wtime();
    #pragma omp target
    {
      #pragma omp teams distribute
      for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] += b[i] * c[i];
        double prev = aa[i];
        for (int64_t j = 1; j < LEN_2D; ++j) {
          prev = bb[j * LEN_2D + i] * a[i] + prev;
          aa[j * LEN_2D + i] = prev;
        }
      }
    }
    double t2 = omp_get_wtime();
    #pragma omp target exit data map(from: a[0:LEN_2D]) map(from: aa[0:N2])
    double t3 = omp_get_wtime();
    // transfer microbenchmark: 1 GiB scratch
    double *s = malloc(512 << 20);
    double u0 = omp_get_wtime();
    #pragma omp target enter data map(to: s[0:(1 << 26)])
    double u1 = omp_get_wtime();
    #pragma omp target exit data map(from: s[0:(1 << 26)])
    double u2 = omp_get_wtime();
    free(s);
    const double g2 = 0.5; // GiB transferred (512 MiB)
    printf("SPLIT N=%lld h2d=%.3f kern=%.3f d2h=%.3f | h2d_gbps=%.1f d2h_gbps=%.1f\n", (long long)LEN_2D,
           (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, g2 / (u1 - u0), g2 / (u2 - u1));
    fflush(stdout);
    return;
  }
  run_body(a, aa, b, bb, c, LEN_2D);
}
