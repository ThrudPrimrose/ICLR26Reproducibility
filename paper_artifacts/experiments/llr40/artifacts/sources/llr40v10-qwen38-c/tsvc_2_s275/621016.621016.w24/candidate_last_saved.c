#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <omp.h>

/* TSVC tsvc_2 s275: per-column scan  aa[j,i] = aa[j-1,i] + bb[j,i]*cc[j,i]
 * (only columns with aa[0,i] > 0). Columns are independent: we parallelize
 * over 8-column groups and interleave the 8 recurrences for ILP. Each group
 * streams 3 contiguous 64B lines per row. Inactive columns are computed but
 * their original value is stored back, leaving them untouched. */
static inline int64_t count_cpus(void) {
#if defined(__linux__)
  cpu_set_t cs;
  if (sched_getaffinity(0, sizeof(cs), &cs) == 0) {
    int64_t c = 0;
    for (int i = 0; i < CPU_SETSIZE; i++) if (CPU_ISSET(i, &cs)) c++;
    if (c > 0) return c;
  }
#endif
  return omp_get_max_threads();
}

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  const int64_t ng8 = (n >> 3) << 3;

  int64_t nthreads = omp_get_max_threads();
  int64_t nc = count_cpus();
  if (nc > 0 && nc < nthreads) nthreads = nc;
  if (nthreads < 1) nthreads = 1;
  omp_set_num_threads((int)nthreads);

  #pragma omp parallel for schedule(static, 1)
  for (int64_t base = 0; base < ng8; base += 8) {
    double s0 = aa[base + 0], s1 = aa[base + 1], s2 = aa[base + 2], s3 = aa[base + 3];
    double s4 = aa[base + 4], s5 = aa[base + 5], s6 = aa[base + 6], s7 = aa[base + 7];
    const int c0 = s0 > 0.0, c1 = s1 > 0.0, c2 = s2 > 0.0, c3 = s3 > 0.0;
    const int c4 = s4 > 0.0, c5 = s5 > 0.0, c6 = s6 > 0.0, c7 = s7 > 0.0;

    const double *rb = bb + base;
    const double *rc = cc + base;
    double *ra = aa + base;

    for (int64_t j = 1; j < n; j++) {
      const int64_t r = j * n;
      s0 += rb[r + 0] * rc[r + 0];
      s1 += rb[r + 1] * rc[r + 1];
      s2 += rb[r + 2] * rc[r + 2];
      s3 += rb[r + 3] * rc[r + 3];
      s4 += rb[r + 4] * rc[r + 4];
      s5 += rb[r + 5] * rc[r + 5];
      s6 += rb[r + 6] * rc[r + 6];
      s7 += rb[r + 7] * rc[r + 7];
      ra[r + 0] = c0 ? s0 : ra[r + 0];
      ra[r + 1] = c1 ? s1 : ra[r + 1];
      ra[r + 2] = c2 ? s2 : ra[r + 2];
      ra[r + 3] = c3 ? s3 : ra[r + 3];
      ra[r + 4] = c4 ? s4 : ra[r + 4];
      ra[r + 5] = c5 ? s5 : ra[r + 5];
      ra[r + 6] = c6 ? s6 : ra[r + 6];
      ra[r + 7] = c7 ? s7 : ra[r + 7];
    }
  }

  /* tail columns: fewer than 8 remain */
  for (int64_t i = ng8; i < n; i++) {
    if (aa[i] > 0.0) {
      for (int64_t j = 1; j < n; j++)
        aa[j * n + i] = aa[(j - 1) * n + i] + bb[j * n + i] * cc[j * n + i];
    }
  }
}
