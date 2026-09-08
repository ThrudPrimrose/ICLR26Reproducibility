/* Optimized version of tsvc_2_s319 kernel with OpenMP parallelism and vectorization.
 * Computes a[i] = c[i] + d[i]; b[i] = c[i] + e[i];
 * Sums all a[i] and b[i] values into `sum`, then stores sum into b[0].
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    double sum = 0.0;

    // Parallelize the outer loop. Use reduction to safely accumulate sum across threads.
    // Combine with SIMD for vectorization.
#pragma omp parallel for simd reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ci = c[i];
        double di = d[i];
        double ei = e[i];
        double ai = ci + di;
        double bi = ci + ei;
        a[i] = ai;
        b[i] = bi;
        sum += ai + bi;
    }
    // Overwrite b[0] with the accumulated sum.
    b[0] = sum;
}
