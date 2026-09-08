/* Optimized version of TSVC tsvc_2_s3110 microkernel.
 * Finds the maximum element in a LEN_2D x LEN_2D matrix `aa` and its coordinates, then
 * writes checksum = max + xindex + yindex to `bb[0]`.
 * Uses OpenMP parallel reduction with a custom reduction type to parallelize the search.
 */

#include <stdint.h>
#include <float.h>
#include <math.h>
#include <omp.h>

static inline int64_t idx(int64_t i, int64_t j, int64_t n) { return i * n + j; }

typedef struct {
    double maxv;
    int64_t xindex;
    int64_t yindex;
} maxloc_t;

/* Custom OpenMP reduction that keeps the maximum value and its indices. */
#pragma omp declare reduction(maxloc : maxloc_t : \
    omp_out = omp_in.maxv > omp_out.maxv ? omp_in : omp_out) \
    initializer(omp_priv = (maxloc_t){ .maxv = -DBL_MAX, .xindex = -1, .yindex = -1 })

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    maxloc_t best = { .maxv = -DBL_MAX, .xindex = -1, .yindex = -1 };
    #pragma omp parallel for collapse(2) reduction(maxloc:best) schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        for (int64_t j = 0; j < LEN_2D; ++j) {
            double v = aa[idx(i, j, LEN_2D)];
            if (v > best.maxv) {
                best.maxv = v;
                best.xindex = i;
                best.yindex = j;
            }
        }
    }
    double chksum = best.maxv + (double)best.xindex + (double)best.yindex;
    bb[0] = chksum;
}

