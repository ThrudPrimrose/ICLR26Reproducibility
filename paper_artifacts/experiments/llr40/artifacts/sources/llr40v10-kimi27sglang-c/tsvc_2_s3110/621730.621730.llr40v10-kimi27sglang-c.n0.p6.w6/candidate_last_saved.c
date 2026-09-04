#include <stdint.h>
#include <float.h>
#include <omp.h>

typedef struct {
    double v;
    int64_t i;
} vp;

#pragma omp declare reduction(maxloc : vp : \
    omp_out = (omp_in.v > omp_out.v) ? omp_in : omp_out) \
    initializer(omp_priv = {-(DBL_MAX), INT64_MAX})

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t n = LEN_2D * LEN_2D;
    vp loc = {-(DBL_MAX), INT64_MAX};

    #pragma omp parallel for reduction(maxloc: loc)
    for (int64_t k = 0; k < n; ++k) {
        double val = aa[k];
        if (val > loc.v) {
            loc.v = val;
            loc.i = k;
        }
    }

    int64_t xindex = loc.i / LEN_2D;
    int64_t yindex = loc.i % LEN_2D;
    bb[0] = loc.v + (double)xindex + (double)yindex;
}
