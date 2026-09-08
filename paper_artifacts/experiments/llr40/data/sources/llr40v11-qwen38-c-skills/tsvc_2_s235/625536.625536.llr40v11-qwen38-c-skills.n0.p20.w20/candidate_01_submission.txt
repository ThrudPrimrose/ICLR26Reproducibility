/* Optimized TSVC tsvc_2 s235 (fp64, single invocation).
 *
 * Dependence analysis of the reference:
 *   a[i] += b[i]*c[i]
 *   aa[j*N+i] = aa[(j-1)*N+i] + bb[j*N+i]*a[i]
 *   dependence vectors in (i,j): only (0,+1) -> j is the serial chain,
 *   i is completely free.  The reference walks i outer / j inner, i.e.
 *   down a column of a row-major matrix (stride N).  Interchanging
 *   (legal: (0,+1) -> (+1,0)) makes every access unit stride and exposes
 *   the i axis for threading.
 *
 * Each thread owns a contiguous column span; the span (a few KB..tens of
 * KB) stays in L1/L2 across j iterations, so the read of aa[j-1] is a
 * cache hit and DRAM sees bb (read) + aa (writeback) = 16*N^2 bytes.
 */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n <= 0) return;

  /* Tiny problem: skip the OpenMP fork entirely. */
  if (n <= 128) {
    #pragma omp simd
    for (int64_t i = 0; i < n; i++)
      a[i] += b[i] * c[i];
    for (int64_t j = 1; j < n; j++) {
      const double *prev = aa + (j - 1) * n;
      const double *rbb  = bb + j * n;
      double *wr         = aa + j * n;
      #pragma omp simd
      for (int64_t i = 0; i < n; i++)
        wr[i] = prev[i] + rbb[i] * a[i];
    }
    return;
  }

  const int64_t nthreads = (int64_t)omp_get_max_threads();
  const int64_t nspan = 20;                       /* spans per thread */
  const int64_t total = nthreads * nspan;
  int64_t chunk = (n + total - 1) / total;
  chunk = (chunk + 3) & ~3LL;                      /* multiple of 4 doubles */
  if (chunk < 4) chunk = 4;
  const int64_t nblocks = (n + chunk - 1) / chunk;

  #pragma omp parallel for schedule(static)
  for (int64_t t = 0; t < nblocks; t++) {
    const int64_t i0 = t * chunk;
    const int64_t i1 = i0 + chunk < n ? i0 + chunk : n;
    const int64_t len = i1 - i0;

    #pragma omp simd
    for (int64_t i = 0; i < len; i++)
      a[i0 + i] += b[i0 + i] * c[i0 + i];

    for (int64_t j = 1; j < n; j++) {
      const double *prev = aa + (j - 1) * n + i0;
      const double *rbb  = bb + j * n + i0;
      double *wr         = aa + j * n + i0;
      #pragma omp simd
      for (int64_t i = 0; i < len; i++)
        wr[i] = prev[i] + rbb[i] * a[i0 + i];
    }
  }
}
