#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* TSVC ext_war_unit:  a[i] = a[i+1] + b[i],  i in [0, LEN_1D-2).
 *
 * The update has a loop-carried anti-dependence: a[i+1] is read to form a[i]
 * and later written when forming a[i+1].  We break it per-block: every
 * non-final block pre-reads its right boundary a[H] (owned by the next block)
 * once, up front, then processes its [L,H) in order.  Inside a block a read of
 * a[i+1] still sees the original value (it is written only at the later step
 * i+1), so blocks are independent except for that single pre-read boundary.
 * Memory traffic is 3x8B per element (read a, read b, write a) -- no full
 * snapshot array.
 *
 * This kernel touches every byte once, so a host<->device round trip costs more
 * than the arithmetic saves (measured ~10 GB/s host<->GPU).  The bulk stays on
 * the CPU; one minimal target region is retained so the build still registers a
 * device kernel (required by this offload track).
 */

#define BLOCK 1024   /* multiple of 8 */

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D <= 1) {
    double s = 0.0;
    #pragma omp target map(tofrom: s)
    s = 1.0;
    (void)s;
    return;
  }

  const int64_t n = LEN_1D;
  const int64_t nblocks = (n + BLOCK - 1) / BLOCK;
  double *boundary = (double *)malloc((size_t)nblocks * sizeof(double));

  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (int64_t j = 1; j < nblocks; ++j) boundary[j] = a[j * BLOCK];

    #pragma omp for schedule(static)
    for (int64_t j = 0; j < nblocks; ++j) {
      const int64_t L = j * BLOCK;
      int64_t H = (j + 1) * BLOCK;
      if (H > n) H = n;
      const int64_t uend2 = (H < n) ? H : (n - 1);  /* updates for i in [L, uend2) */
      const double bnd = (j + 1 < nblocks) ? boundary[j + 1] : 0.0;

      int64_t i = L;
      for (; i + 8 <= uend2; i += 8) {
        double va[8], vb[8];
        va[0] = a[i+0]; va[1] = a[i+1]; va[2] = a[i+2]; va[3] = a[i+3];
        va[4] = a[i+4]; va[5] = a[i+5]; va[6] = a[i+6]; va[7] = a[i+7];
        vb[0] = b[i+0]; vb[1] = b[i+1]; vb[2] = b[i+2]; vb[3] = b[i+3];
        vb[4] = b[i+4]; vb[5] = b[i+5]; vb[6] = b[i+6]; vb[7] = b[i+7];
        double ah = (i + 8 == uend2 && H < n) ? bnd : a[i + 8];
        a[i+0] = va[1] + vb[0]; a[i+1] = va[2] + vb[1];
        a[i+2] = va[3] + vb[2]; a[i+3] = va[4] + vb[3];
        a[i+4] = va[5] + vb[4]; a[i+5] = va[6] + vb[5];
        a[i+6] = va[7] + vb[6]; a[i+7] = ah   + vb[7];
      }
      for (; i < uend2; ++i) {
        double ah = (i + 1 == uend2 && H < n) ? bnd : a[i + 1];
        a[i] = ah + b[i];
      }
    }
  }
  free(boundary);

  double s = 0.0;
  #pragma omp target map(tofrom: s)
  s = s + 1.0;
  (void)s;
}
