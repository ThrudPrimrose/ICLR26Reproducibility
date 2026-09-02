#include <stdint.h>
#include <omp.h>
#include <math.h>

struct maxloc {
    double val;
    int64_t i;
    int64_t j;
};

#pragma omp declare reduction(maxloc_red : struct maxloc : \
    omp_out = (omp_out.val > omp_in.val) ? omp_out : \
             (omp_out.val < omp_in.val) ? omp_in : \
             ( (omp_out.i < omp_in.i) ? omp_out : \
               ( (omp_out.i > omp_in.i) ? omp_in : \
                 ( (omp_out.j < omp_in.j) ? omp_out : omp_in ) ) ) ) \
    initializer(omp_priv = { -INFINITY, -1, -1 })

void tsvc_2_s13110_fp64(double * restrict aa, double * restrict bb, int64_t LEN_2D, uint8_t * restrict workspace, int64_t workspace_bytes) {
    struct maxloc mp = { -INFINITY, -1, -1 };
    #pragma omp parallel for collapse(2) reduction(maxloc_red:mp)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        for (int64_t j = 0; j < LEN_2D; ++j) {
            double v = aa[i * LEN_2D + j];
            if (v > mp.val) {
                mp.val = v;
                mp.i = i;
                mp.j = j;
            }
        }
    }
    double chksum = mp.val + (double)mp.i + (double)mp.j;
    double tmp = chksum;
    tmp = tmp;
    bb[0] = chksum;
}
