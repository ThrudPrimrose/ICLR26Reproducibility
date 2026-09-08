#include <stdint.h>
#include <omp.h>

/* tsvc_2 s231: aa[j,i] = aa[j-1,i] + bb[j,i]  (j=1..LEN-1, all i).
 * Dependency is along j (row dir) for fixed i; the i (column) direction is
 * independent and contiguous in memory.  Each thread owns a contiguous block
 * of columns and, per row j, does a vector add across that block:
 *   - vectorizes (contiguous i),
 *   - parallelizes across columns (aggregate DRAM bandwidth; 24 threads = one
 *     NUMA node of cores is the sweet spot),
 *   - bit-exact: each (j,i) is computed exactly as in the reference.
 * The kernel is write-bandwidth bound; prefetching the bb row two rows ahead
 * (L1 locality) keeps the read stream fed and avoids write stalls.
 */
void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  #pragma omp parallel
  {
    const int nt  = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t base = LEN_2D / nt;
    const int64_t rem  = LEN_2D % nt;
    const int64_t i0   = tid * base + (tid < rem ? tid : rem);
    const int64_t i1   = i0 + base + (tid < rem ? 1 : 0);

    for (int64_t j = 1; j < LEN_2D; ++j) {
      double *restrict cur  = aa + j * LEN_2D;
      const double *restrict prev = aa + (j - 1) * LEN_2D;
      const double *restrict brow = bb + j * LEN_2D;
      if (j + 2 < LEN_2D) {
        const double *pb = bb + (j + 2) * LEN_2D;
        for (int64_t p = i0; p < i1; p += 2)
          __builtin_prefetch(pb + p, 0, 1);
      }
      for (int64_t i = i0; i < i1; ++i)
        cur[i] = prev[i] + brow[i];
    }
  }
}
