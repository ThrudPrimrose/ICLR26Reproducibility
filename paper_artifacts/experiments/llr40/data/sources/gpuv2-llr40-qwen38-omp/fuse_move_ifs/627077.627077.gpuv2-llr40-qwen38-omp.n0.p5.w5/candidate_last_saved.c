/* TSVC fuse_move_ifs -- OpenMP target offload version.
 * Two guarded loop nests:
 *   pass A: a[i,j] = 2*src[i,j] for rows with cond[i] > 0  (other rows untouched)
 *   pass B: b[i,j] = src[i,j] + 1 for all rows, only if K > 0 (b untouched otherwise)
 * Data flows host<->device via explicit maps (the harness moves nothing).
 */
#include <stdint.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  const int64_t N = LEN_2D * LEN_2D;

  /* Bring src in once; both passes read it. */
  #pragma omp target data map(to: src[0:N])
  {
    /* a must round-trip (tofrom): rows with cond<=0 are never written on device,
     * and the host copy must keep its original values there. */
    #pragma omp target map(to: cond[0:LEN_2D]) map(tofrom: a[0:N])
    {
      #pragma omp teams distribute parallel for collapse(2)
      for (int64_t i = 0; i < LEN_2D; ++i)
        for (int64_t j = 0; j < LEN_2D; ++j)
          if (cond[i] > 0.0)
            a[i * LEN_2D + j] = src[i * LEN_2D + j] * 2.0;
    }
    if (K > 0) {
      /* b is fully written when K>0, so a plain (from) map of fresh device
       * memory is safe; when K<=0 we must not touch host b at all. */
      #pragma omp target map(from: b[0:N])
      {
        #pragma omp teams distribute parallel for collapse(2)
        for (int64_t i = 0; i < LEN_2D; ++i)
          for (int64_t j = 0; j < LEN_2D; ++j)
            b[i * LEN_2D + j] = src[i * LEN_2D + j] + 1.0;
      }
    }
  }
}
