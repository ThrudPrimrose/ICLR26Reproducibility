#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb,
                      const double *restrict cc, const int64_t LEN_2D) {
  const int64_t n = LEN_2D - 8; /* active rows/cols */
  if (n <= 0) return;

  int nt;
  #pragma omp parallel
  #pragma omp single
  nt = omp_get_num_threads();
  if (nt < 1) nt = 1;

  /* aa[j][i] = aa[j-1][i] + cc[j][i]: recurrence along j for each column i;
     columns independent -> partition i across threads, vectorize inner i loop. */
  #pragma omp parallel num_threads(nt)
  {
    const int tid = omp_get_thread_num();
    const int64_t base = n / nt, rem = n % nt;
    const int64_t i0 = 8 + tid * base + (tid < rem ? tid : rem);
    const int64_t i1 = i0 + base + (tid < rem ? 1 : 0);
    for (int64_t j = 8; j < LEN_2D; ++j) {
      double *row = aa + j * LEN_2D;
      const double *prev = aa + (j - 1) * LEN_2D;
      const double *c = cc + j * LEN_2D;
      for (int64_t i = i0; i < i1; ++i)
        row[i] = prev[i] + c[i];
    }
  }

  /* bb[j][i] = bb[j][i-1] + cc[j][i]: recurrence along i for each row j;
     rows independent -> partition j across threads. */
  #pragma omp parallel for schedule(static)
  for (int64_t j = 8; j < LEN_2D; ++j) {
    double *row = bb + j * LEN_2D;
    const double *c = cc + j * LEN_2D;
    for (int64_t i = 8; i < LEN_2D; ++i)
      row[i] = row[i - 1] + c[i];
  }
}
