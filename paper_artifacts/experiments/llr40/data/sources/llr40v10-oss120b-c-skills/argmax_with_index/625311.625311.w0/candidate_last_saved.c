/* Optimized argmax with index for double precision arrays.
   Uses OpenMP parallel reduction with a custom reduction type to find the maximum value
   and its earliest occurrence index.
   The reference signature is:
   void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                               double *restrict out_value, const int64_t LEN_1D);

   This implementation is safe under the C23 ABI, uses restrict qualifiers for aliasing,
   and relies only on the standard library and OpenMP.
*/

#include <stdint.h>
#include <omp.h>
#include <math.h>

/* Struct to hold a value and its index. */
typedef struct {
    double val;
    int64_t idx;
} maxloc_t;

/* Declare a user-defined reduction that picks the larger value.
   In case of equal values the smaller index (earlier occurrence) wins. */
#pragma omp declare reduction(maxloc : maxloc_t : \
    omp_out = (omp_in.val > omp_out.val) ? omp_in : \
    (omp_in.val < omp_out.val) ? omp_out : \
    (omp_in.idx < omp_out.idx) ? omp_in : omp_out) \
    initializer(omp_priv = (maxloc_t){ -INFINITY, INT64_MAX })

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    /* Guard against empty input – the reference assumes LEN_1D >= 1. */
    if (LEN_1D <= 0) {
        out_value[0] = -INFINITY;
        out_index[0] = -1;
        return;
    }

    maxloc_t result = { -INFINITY, INT64_MAX };

    /* Parallel loop with static schedule for contiguous chunks per thread. */
    #pragma omp parallel for reduction(maxloc:result) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = a[i];
        if (v > result.val) {
            result.val = v;
            result.idx = i;
        } else if (v == result.val && i < result.idx) {
            /* Tie – keep the smallest index. */
            result.idx = i;
        }
    }

    out_value[0] = result.val;
    out_index[0] = result.idx;
}

