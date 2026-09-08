/* Optimized version of tsvc_2_s2710_fp64 kernel.
   Moves loop-invariant conditionals out of the loop and adds OpenMP parallelism.
*/
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
    bool len_gt_10 = LEN_1D > 10;
    if (len_gt_10) {
        #pragma omp target data map(tofrom: a[0:LEN_1D], b[0:LEN_1D], c[0:LEN_1D]) map(to: d[0:LEN_1D], e[0:LEN_1D], x[0:1])
        {
            #pragma omp target teams distribute parallel for simd
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (a[i] > b[i]) {
                a[i] += b[i] * d[i];
                c[i] += d[i] * d[i];
            } else {
                b[i] = a[i] + e[i] * e[i];
                if (x[0] > 0.0) {
                    c[i] = a[i] + d[i] * d[i];
                } else {
                    c[i] += e[i] * e[i];
                }
            }
        }
        }
    } else {
        /* Small problem size; parallel overhead may dominate.
           Use a simple sequential loop.
        */
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (a[i] > b[i]) {
                a[i] += b[i] * d[i];
                c[i] = d[i] * e[i] + 1.0;
            } else {
                b[i] = a[i] + e[i] * e[i];
                if (x[0] > 0.0) {
                    c[i] = a[i] + d[i] * d[i];
                } else {
                    c[i] += e[i] * e[i];
                }
            }
        }
    }
}
