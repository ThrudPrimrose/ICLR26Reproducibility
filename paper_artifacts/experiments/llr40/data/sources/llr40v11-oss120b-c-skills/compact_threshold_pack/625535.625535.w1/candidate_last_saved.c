#include <stdint.h>

/*
 * Reference implementation for the ``compact_threshold_pack`` kernel.
 *
 * Input arrays:
 *   src   – double array of length LEN_1D
 *   weight – double array of length LEN_1D
 *
 * Output arrays:
 *   packed   – double array of length LEN_1D (only the first `n` elements are written)
 *   out_count – int64_t array of length 1, where out_count[0] receives the number of packed elements.
 *
 * For each index i in [0, LEN_1D): if src[i] > 0.0, the product src[i] * weight[i] is stored
 * into packed at the next free position. The order of survivors is preserved. The final count
 * of elements written is stored in out_count[0].
 */

void compact_threshold_pack_fp64(const double *restrict src,
                                 const double *restrict weight,
                                 int64_t *restrict out_count,
                                 double *restrict packed,
                                 const int64_t LEN_1D) {
    const double *src_ptr = src;
    const double *weight_ptr = weight;
    double *packed_ptr = packed;
    int64_t count = 0;

    for (int64_t i = 0; i < LEN_1D; ++i) {
        double s = *src_ptr++;
        double w = *weight_ptr++;
        if (s > 0.0) {
            *packed_ptr++ = s * w;
            ++count;
        }
    }
    out_count[0] = count;
}
