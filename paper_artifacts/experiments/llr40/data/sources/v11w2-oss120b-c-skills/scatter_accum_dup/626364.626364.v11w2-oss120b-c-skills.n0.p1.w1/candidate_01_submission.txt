#include <stdint.h>
#include <omp.h>

/*
 * Kernel: scatter_accum_dup
 * Description: Perform indexed accumulate with possible duplicate indices.
 *   bins[ip[i]] += src[i];
 *   The ip array may contain duplicate indices, requiring atomic updates for correctness when parallelized.
 *   The function works on double-precision values.
 *
 * Arguments:
 *   bins   - array of length LEN_1D, holds the destination values (read-modify-write).
 *   src    - source array of length LEN_1D.
 *   ip     - index array of length LEN_1D, values in [0, LEN_1D-1].
 *   LEN_1D - number of elements.
 */

void scatter_accum_dup_fp64(double *restrict bins,
                       const int32_t *restrict ip,
                       const double *restrict src,
                       const int64_t LEN_1D,
                       uint8_t *restrict workspace,
                       const int64_t workspace_bytes) {
    #pragma omp parallel for schedule(static) default(none) shared(bins, src, ip, LEN_1D)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int32_t idx = ip[i];
        #pragma omp atomic
        bins[idx] += src[i];
    }
}

