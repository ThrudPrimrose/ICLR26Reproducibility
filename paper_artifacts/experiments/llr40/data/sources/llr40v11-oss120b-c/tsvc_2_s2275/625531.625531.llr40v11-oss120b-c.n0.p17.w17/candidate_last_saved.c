/* Optimized version of tsvc_2_s2275_fp64 kernel.
   Rewrites the original double-nested loop as two independent flat loops
   to improve cache locality and vectorization, and adds OpenMP parallelism.
*/

#include <stdint.h>

void tsvc_2_s2275_fp64(double *restrict a,
                       double *restrict aa,
                       const double *restrict b,
                       const double *restrict bb,
                       const double *restrict c,
                       const double *restrict cc,
                       const double *restrict d,
                       const int64_t LEN_2D) {
    // Update aa = aa + bb * cc elementwise over LEN_2D * LEN_2D entries.
    const int64_t N = LEN_2D;
    const int64_t total = N * N;

    #pragma omp parallel for simd schedule(static)
    for (int64_t k = 0; k < total; ++k) {
        aa[k] = aa[k] + bb[k] * cc[k];
    }

    // Update a = b + c * d for each element.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }
}
