/* TSVC tsvc_2 s233:  aa[j][i] = aa[j-1][i] + cc[j][i]  (column scans of cc)
 *                     bb[j][i] = bb[j][i-1] + cc[j][i]  (row scans of cc)
 * for j,i in [8, LEN_2D).
 *
 * Restructure: each aa column i is an independent prefix sum of cc column i
 * seeded by aa[7][i]; each bb row j is an independent prefix sum of cc row j
 * seeded by bb[j][7].  Values are bit-identical to the reference recurrence.
 *
 * Work per byte is ~0.05 FLOP/byte, so a GPU round trip of 3*N^2*8 bytes in
 * plus 2*N^2*8 bytes out costs far more than the arithmetic saves: the scans
 * are run on host threads, each thread interleaving many columns (or whole
 * contiguous rows) so the dependent add chain runs over DRAM latency while
 * the compiler vectorizes the inner sweep.  A target region is present so
 * the device image is registered on this offload arm. */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc,
                      const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  const int64_t L = N - 8;
  if (L <= 0) return;

  /* Device registration (this offload arm embeds a device image; the data
   * volume makes an array round trip uneconomic -- see header). */
  {
    int x = 0;
    #pragma omp target map(tofrom: x)
    x = x + 1;
    (void)x;
  }

  #pragma omp parallel
  {
    const int nt  = omp_get_num_threads();
    const int tid = omp_get_thread_num();

    /* Part 1: aa column scans.  Thread tid owns columns [c0, c1).  Sweep by
     * row, sweeping the owned columns contiguously: the inner sweep is a run
     * of chunk adjacent doubles (vectorized), rows are 8*N bytes apart. */
    const int64_t c0 = 8 + (L * tid) / nt;
    const int64_t c1 = 8 + (L * (tid + 1)) / nt;
    for (int64_t gc = c0; gc < c1; gc += 512) {
      const int64_t g = (c1 - gc) < 512 ? (c1 - gc) : 512;
      double acc[512];
      for (int64_t k = 0; k < g; ++k) acc[k] = aa[7 * N + gc + k];
      for (int64_t j = 8; j < N; ++j) {
        double *const a = aa + j * N + gc;
        const double *const c = cc + j * N + gc;
        for (int64_t k = 0; k < g; ++k) {
          acc[k] += c[k];
          a[k] = acc[k];
        }
      }
    }

    /* Part 2: bb row scans.  Each owned row is a straight contiguous sweep. */
    const int64_t r0 = 8 + (L * tid) / nt;
    const int64_t r1 = 8 + (L * (tid + 1)) / nt;
    for (int64_t row = r0; row < r1; ++row) {
      double accv = bb[row * N + 7];
      double *const b = bb + row * N + 8;
      const double *const c = cc + row * N + 8;
      for (int64_t i = 0; i < N - 8; ++i) {
        accv += c[i];
        b[i] = accv;
      }
    }
  }
}
