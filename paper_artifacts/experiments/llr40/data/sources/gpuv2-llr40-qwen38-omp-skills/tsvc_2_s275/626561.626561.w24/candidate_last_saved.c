#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  const int64_t m = n - 1;                 // scan elements per column (j = 1..n-1)
  if (m <= 0) return;

  const int64_t B = 128;                    // chunk size (rows per chunk)
  const int64_t C = (m + B - 1) / B;        // number of chunks
  const int64_t nC = n * C;

  // device-only scratch: tot[c*n+i] = chunk-c total of column i; off[c*n+i] = exclusive prefix
  double *tot = (double *)malloc((size_t)nC * sizeof(double));
  double *off = (double *)malloc((size_t)nC * sizeof(double));

  #pragma omp target data \
      map(tofrom: aa[0:n*n]) \
      map(to: bb[0:n*n]) \
      map(to: cc[0:n*n]) \
      map(alloc: tot[0:nC]) \
      map(alloc: off[0:nC])
  {
    // Phase 1: per (chunk c, column i), running partial sum of d = bb*cc over the chunk.
    // Partial stored in-place in aa (active columns only); chunk total in tot.
    #pragma omp target teams distribute parallel for collapse(2)
    for (int64_t c = 0; c < C; c++)
      for (int64_t i = 0; i < n; i++) {
        const int64_t r0 = c * B;
        if (r0 >= m) continue;
        const int64_t r1 = (r0 + B < m) ? r0 + B : m;
        if (aa[i] > 0.0) {
          double acc = 0.0;
          for (int64_t k = r0; k < r1; k++) {
            const int64_t idx = (k + 1) * n + i;
            acc += bb[idx] * cc[idx];
            aa[idx] = acc;                  // local partial (chunk-relative)
          }
          tot[c * n + i] = acc;
        } else {
          tot[c * n + i] = 0.0;
        }
      }

    // Phase 2: exclusive prefix over chunk totals, per column (parallel over i, serial over c).
    #pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < n; i++) {
      double run = 0.0;
      for (int64_t c = 0; c < C; c++) {
        off[c * n + i] = run;
        run += tot[c * n + i];
      }
    }

    // Phase 3: add row0 + chunk offset to the partials -> final aa.
    #pragma omp target teams distribute parallel for collapse(2)
    for (int64_t c = 0; c < C; c++)
      for (int64_t i = 0; i < n; i++) {
        const int64_t r0 = c * B;
        if (r0 >= m) continue;
        const int64_t r1 = (r0 + B < m) ? r0 + B : m;
        if (aa[i] > 0.0) {
          const double base = aa[i] + off[c * n + i];
          for (int64_t k = r0; k < r1; k++) {
            const int64_t idx = (k + 1) * n + i;
            aa[idx] = base + aa[idx];
          }
        }
      }
  }
  free(tot);
  free(off);
}
