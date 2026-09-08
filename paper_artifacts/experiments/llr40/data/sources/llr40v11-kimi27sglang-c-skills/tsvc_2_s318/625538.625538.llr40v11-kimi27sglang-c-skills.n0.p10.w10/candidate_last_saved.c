#include <math.h>
#include <stdint.h>
#include <float.h>
#include <omp.h>

typedef struct {
    double v;
    int64_t i;
} argmax_pair_t;

#pragma omp declare reduction(argmax : argmax_pair_t : \
    omp_out = (omp_out.v > omp_in.v) ? omp_out : \
              ((omp_out.v < omp_in.v) ? omp_in : \
               ((omp_out.i < omp_in.i) ? omp_out : omp_in))) \
    initializer(omp_priv = {-HUGE_VAL, INT64_MAX})

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
    const int64_t n = LEN_1D;
    argmax_pair_t best = { fabs(a[0]), 0 };

    if (n > 1) {
        #pragma omp parallel for reduction(argmax : best) schedule(static) if(n > 2048)
        for (int64_t i = 1; i < n; ++i) {
            double v = fabs(a[i * inc]);
            if (v > best.v) {
                best.v = v;
                best.i = i;
            }
        }
    }

    result[0] = best.v + (double)best.i;
}
