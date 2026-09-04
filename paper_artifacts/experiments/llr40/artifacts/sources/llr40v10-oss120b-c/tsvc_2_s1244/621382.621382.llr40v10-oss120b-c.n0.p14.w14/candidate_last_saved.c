/* Optimized version of TSVC tsvc_2_s1244 kernel.
   Uses a temporary copy of the original a array to break the dependency between a[i] and a[i+1].
   The copy and the computation of new a values are performed in a single parallel region to reduce
   thread creation overhead. This enables parallelism and vectorization while preserving exact results.
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
    // Allocate buffer to hold original a values.
    double *old_a = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!old_a) {
        // Allocation failure: abort.
        return;
    }

    #pragma omp parallel
    {
        // Copy original a to old_a and compute new a values (for indices < LEN_1D-1).
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            old_a[i] = a[i];
            if (i < LEN_1D - 1) {
                double bi = b[i];
                double ci = c[i];
                a[i] = bi + ci * ci + bi * bi + ci;
            }
        }

        // Compute d[i] using the original a[i+1] values stored in old_a.
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            d[i] = a[i] + old_a[i + 1];
        }
    }

    free(old_a);
}
