/* Optimized version of TSVC tsvc_2 kernel s3110 (fp64).
 * Finds the maximum element and its coordinates in a LEN_2D x LEN_2D matrix.
 * Uses OpenMP parallel reduction with a custom reduction type to exploit
 * multi-core parallelism and SIMD vectorization.
 */

#include <stdint.h>
#include <float.h>

/* Custom reduction type that stores the current maximum value and the flat
 * index (i*LEN_2D + j) where it occurs. In case of ties we keep the smallest
 * index, which corresponds to the first occurrence in row‑major order.
 */
struct maxloc {
    double v;      /* current maximum value */
    int64_t idx;   /* flat index of the maximum */
};

/* Reduction operator: keep the larger value. If values are equal, keep the
 * smaller index (earlier occurrence).
 */
#pragma omp declare reduction(maxloc_red : struct maxloc : \
    omp_out = (omp_in.v > omp_out.v) ? omp_in : \
              ((omp_in.v < omp_out.v) ? omp_out : \
               ((omp_in.idx < omp_out.idx) ? omp_in : omp_out)) ) \
    initializer(omp_priv = (struct maxloc){ .v = -DBL_MAX, .idx = -1 })

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t total = LEN_2D * LEN_2D;
    struct maxloc best = { .v = -DBL_MAX, .idx = -1 };

    /* Parallel reduction over the flat array. */
    #pragma omp parallel for reduction(maxloc_red:best)
    for (int64_t flat = 0; flat < total; ++flat) {
        double v = aa[flat];
        if (v > best.v) {
            best.v = v;
            best.idx = flat;
        } else if (v == best.v && flat < best.idx) {
            /* Tie – keep the earliest index. */
            best.idx = flat;
        }
    }

    int64_t xindex = best.idx / LEN_2D;
    int64_t yindex = best.idx % LEN_2D;
    double chksum = best.v + (double)xindex + (double)yindex;
    bb[0] = chksum;
}
