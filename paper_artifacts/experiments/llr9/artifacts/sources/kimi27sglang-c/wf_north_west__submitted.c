#include <stdint.h>
#include <omp.h>

void wf_north_west_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  const int64_t T = 60;
  const int64_t nt = (n + T - 1) / T;

  omp_set_num_threads(24);
  #pragma omp parallel
  {
    for (int64_t d = 0; d <= 2 * (nt - 1); ++d) {
      int64_t ti0 = (d < nt) ? 0 : d - (nt - 1);
      int64_t ti1 = (d < nt) ? d : nt - 1;
      #pragma omp for schedule(static)
      for (int64_t ti = ti0; ti <= ti1; ++ti) {
        int64_t tj = d - ti;
        int64_t i0 = ti * T; if (i0 < 1) i0 = 1;
        int64_t i1 = (ti + 1) * T; if (i1 > n) i1 = n;
        int64_t j0 = tj * T; if (j0 < 1) j0 = 1;
        int64_t j1 = (tj + 1) * T; if (j1 > n) j1 = n;
        for (int64_t i = i0; i < i1; ++i) {
          for (int64_t j = j0; j < j1; ++j) {
            a[i * n + j] = a[i * n + j] + a[(i - 1) * n + j] + a[i * n + (j - 1)];
          }
        }
      }
    }
  }
}
