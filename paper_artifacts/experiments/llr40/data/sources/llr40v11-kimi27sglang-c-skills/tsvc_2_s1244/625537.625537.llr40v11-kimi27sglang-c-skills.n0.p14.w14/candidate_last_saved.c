#include <stdint.h>
#include <omp.h>

#ifndef BUFSZ
#define BUFSZ 4096
#endif

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
  const int64_t n = LEN_1D - 1;

#pragma omp parallel
  {
    _Alignas(64) double buf[BUFSZ];
    const int tid = omp_get_thread_num();
    const int nt = omp_get_num_threads();
    const int64_t chunk = (n + nt - 1) / nt;
    const int64_t start = tid * chunk;
    const int64_t end = (start + chunk < n) ? start + chunk : n;

    for (int64_t t = start; t < end; t += BUFSZ) {
      const int64_t t_end = (t + BUFSZ < end) ? t + BUFSZ : end;
      const int64_t len = t_end - t;

      for (int64_t j = 0; j < len; j++) {
        buf[j] = a[t + 1 + j];
      }

#pragma omp barrier

      for (int64_t j = 0; j < len; j++) {
        const int64_t i = t + j;
        const double bi = b[i];
        const double ci = c[i];
        a[i] = bi + ci * ci + bi * bi + ci;
        d[i] = a[i] + buf[j];
      }
    }
  }
}
