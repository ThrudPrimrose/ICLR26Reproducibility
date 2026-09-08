#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  const int64_t NN = N * N;

  int probe_dev = -1;
  #pragma omp target map(from: probe_dev)
  probe_dev = !omp_is_initial_device();
  int ndev = omp_get_num_devices();

  double t0 = now_us();
  #pragma omp target enter data map(to: b[0:N]) map(to: bb[0:NN]) map(to: c[0:N]) map(to: a[0:N]) map(to: aa[0:NN])
  double t1 = now_us();
  #pragma omp target
  {
    #pragma omp teams distribute parallel for
    for (int64_t i = 0; i < N; i++) {
      a[i] += b[i] * c[i];
      double s = aa[i];
      for (int64_t j = 1; j < N; j++) {
        s += bb[j * N + i] * a[i];
        aa[j * N + i] = s;
      }
    }
  }
  double t2 = now_us();
  #pragma omp target exit data map(from: a[0:N]) map(from: aa[0:NN])
  double t3 = now_us();
  #pragma omp target exit data map(delete: b[0:N]) map(delete: bb[0:NN]) map(delete: c[0:N])

  printf("LEN_2D=%lld ndev=%d dev=%d enter=%.1fus kernel=%.1fus exit=%.1fus total=%.1fus\n",
         (long long)N, ndev, probe_dev, t1 - t0, t2 - t1, t3 - t2, t3 - t0);
  fflush(stdout);
}
