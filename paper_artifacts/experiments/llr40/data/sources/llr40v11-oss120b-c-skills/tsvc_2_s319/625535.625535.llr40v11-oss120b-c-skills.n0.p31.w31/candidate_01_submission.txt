/* Optimized version of TSVC tsvc_2_s319 kernel using OpenMP parallelism and vectorization.
   Computes:
    a[i] = c[i] + d[i];
    b[i] = c[i] + e[i];
    sum += a[i] + b[i];
   and stores sum into b[0] after the loop.
   All pointers are restrict-qualified, and LEN_1D is the loop bound. */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = c[i] + d[i];
        a[i] = ai;
        sum += ai;
        double bi = c[i] + e[i];
        b[i] = bi;
        sum += bi;
    }
    // Overwrite b[0] with the accumulated sum.
    b[0] = sum;
}
