#include <math.h>
#include <stdint.h>
#include <float.h>
#include <omp.h>

typedef struct {
    double maxv;
    int64_t index;
} maxpair_t;

#pragma omp declare reduction(maxpair : maxpair_t : \
    omp_out = ((omp_in.maxv > omp_out.maxv) || \
               ((omp_in.maxv == omp_out.maxv) && (omp_in.index < omp_out.index))) \
              ? omp_in : omp_out) \
    initializer(omp_priv = (maxpair_t){-INFINITY, INT64_MAX})

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
    if (LEN_1D <= 0) {
        result[0] = 0.0;
        return;
    }

    const double initial_max = fabs(a[0]);
    maxpair_t p = {initial_max, 0};

    if (LEN_1D > 1) {
        if (LEN_1D < 100000) {
            int64_t k = inc;
            for (int64_t i = 1; i < LEN_1D; ++i) {
                const double v = fabs(a[k]);
                if ((v > p.maxv) || ((v == p.maxv) && (i < p.index))) {
                    p.maxv = v;
                    p.index = i;
                }
                k += inc;
            }
        } else {
            #pragma omp parallel for reduction(maxpair:p) schedule(static)
            for (int64_t i = 1; i < LEN_1D; ++i) {
                const double v = fabs(a[i * inc]);
                if ((v > p.maxv) || ((v == p.maxv) && (i < p.index))) {
                    p.maxv = v;
                    p.index = i;
                }
            }
        }
    }

    result[0] = p.maxv + (double)(p.index);
}
