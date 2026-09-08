/* Optimized argmax with index for double array using OpenMP reduction and optional offloading.
   Finds the maximum value and its first occurrence index in array a of length LEN_1D.
   The result is stored in out_value[0] and out_index[0]. */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>

/* Define a struct to hold value and index for reduction. */
typedef struct {
    double val;
    int64_t idx;
} maxpair_t;

/* Declare a custom reduction that selects the larger value; on ties selects the smaller index. */
#pragma omp declare reduction(maxpair : maxpair_t : \
    omp_out = (omp_in.val > omp_out.val) ? omp_in : \
              (omp_in.val < omp_out.val) ? omp_out : \
              (omp_in.idx < omp_out.idx ? omp_in : omp_out)) \
    initializer(omp_priv = { .val = -INFINITY, .idx = -1 })

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        out_value[0] = -INFINITY;
        out_index[0] = -1;
        return;
    }

    maxpair_t result = { .val = -INFINITY, .idx = -1 };

    /* Offload to GPU if the problem size justifies it. The threshold is heuristic; change if needed. */
    const int64_t offload_threshold = INT64_MAX; // effectively never offload

    #pragma omp target teams distribute parallel for reduction(maxpair:result) \
        map(to: a[0:LEN_1D]) \
        map(tofrom: out_index[0:1], out_value[0:1]) \
        if (LEN_1D > offload_threshold) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        if (ai > result.val) {
            result.val = ai;
            result.idx = i;
        } else if (ai == result.val && i < result.idx) {
            result.idx = i;
        }
    }
    /* Write back the result (both host and device copies are updated via the map clause). */
    out_value[0] = result.val;
    out_index[0] = result.idx;
}

