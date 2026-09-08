/* Optimized version of tsvc_2_s319_fp64 kernel with OpenMP offload.
 * Computes a[i] = c[i] + d[i]; b[i] = c[i] + e[i];
 * Accumulates sum of all a[i] and b[i] into sum, then stores sum in b[0].
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    double sum = 0.0;
    /* Offload the computation to the GPU. All arrays are explicitly mapped.
       The reduction clause computes the sum across all threads and teams and
       returns the result to the host variable `sum`. */
    #pragma omp target teams distribute parallel for simd \
        map(to: c[0:LEN_1D], d[0:LEN_1D], e[0:LEN_1D]) \
        map(tofrom: a[0:LEN_1D], b[0:LEN_1D]) \
        reduction(+:sum)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = c[i] + d[i];
        a[i] = ai;
        sum += ai;
        double bi = c[i] + e[i];
        b[i] = bi;
        sum += bi;
    }
    /* Store the reduction result back to the first element of b on the host. */
    b[0] = sum;
}

