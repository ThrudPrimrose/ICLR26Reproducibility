#include <stdint.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { double val; int64_t idx; } maxpair_t;

#pragma omp declare reduction(maxpair : maxpair_t : omp_out = (omp_out.val > omp_in.val) ? omp_out : ((omp_in.val > omp_out.val) ? omp_in : ((omp_out.idx < omp_in.idx) ? omp_out : omp_in))) initializer(omp_priv = (maxpair_t){ -INFINITY, INT64_MAX })

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value, const int64_t LEN_1D) {
    maxpair_t best = { -INFINITY, INT64_MAX };
    #pragma omp parallel for reduction(maxpair:best)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = a[i];
        if (v > best.val || (v == best.val && i < best.idx)) {
            best.val = v;
            best.idx = i;
        }
    }
    out_value[0] = best.val;
    out_index[0] = best.idx;
}
