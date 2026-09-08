#include <stdint.h>
#include <math.h>
#include <omp.h>

/* Compute the maximum value in a square matrix of size LEN_2D x LEN_2D and
 * return checksum = max + row_index + col_index via bb[0]. */
static inline int64_t idx(int64_t i, int64_t j, int64_t n) {
    return i * n + j;
}

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t N = LEN_2D * LEN_2D;
    double maxv = -INFINITY;

    /* First pass: parallel vectorized reduction to find the maximum value */
    #pragma omp parallel for simd reduction(max:maxv) schedule(static)
    for (int64_t k = 0; k < N; ++k) {
        double v = aa[k];
        if (v > maxv) {
            maxv = v;
        }
    }

    /* Second pass: parallel vectorized reduction to find the first (smallest) linear index of maxv */
    int64_t min_idx = N; // sentinel larger than any valid index
    #pragma omp parallel for simd reduction(min:min_idx) schedule(static)
    for (int64_t k = 0; k < N; ++k) {
        if (aa[k] == maxv) {
            if (k < min_idx) {
                min_idx = k;
            }
        }
    }

    int64_t xindex = min_idx / LEN_2D;
    int64_t yindex = min_idx % LEN_2D;
    double chksum = maxv + (double)xindex + (double)yindex;
    bb[0] = chksum;
}
