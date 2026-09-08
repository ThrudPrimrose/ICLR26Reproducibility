#include <math.h>
#include <stdint.h>
#include <omp.h>

typedef struct {
    double maxv;
    int64_t index;
} loc_t;

#pragma omp declare reduction(argmax : loc_t : \
    omp_out = (omp_out.maxv > omp_in.maxv) ? omp_out : \
              ((omp_out.maxv < omp_in.maxv) ? omp_in : \
               ((omp_out.index < omp_in.index) ? omp_out : omp_in))) \
    initializer(omp_priv = omp_orig)

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
    loc_t loc = {fabs(a[0]), 0};

    #pragma omp parallel for reduction(argmax : loc) schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double v = fabs(a[i * inc]);
        if (v > loc.maxv) {
            loc.maxv = v;
            loc.index = i;
        }
    }

    result[0] = loc.maxv + (double)loc.index;
}
