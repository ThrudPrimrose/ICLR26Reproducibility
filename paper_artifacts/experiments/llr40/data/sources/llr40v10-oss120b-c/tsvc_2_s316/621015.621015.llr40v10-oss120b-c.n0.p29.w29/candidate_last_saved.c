/* Optimized version of TSVC tsvc_2 s316 kernel.
   Computes the minimum of an array of double-precision values.
   Uses OpenMP parallel for with SIMD and reduction(min) to exploit
   multi-core and vector units.
   Original reference: /shared/tasks/tsvc_2_s316/tsvc_2_s316_reference.c */

#include <stdint.h>
#include <float.h>

void tsvc_2_s316_fp64(const double * restrict a, double * restrict result, const int64_t LEN_1D) {
    double min_val = DBL_MAX;
    /*
       OpenMP parallel for with simd and reduction(min) allows both
       multi-threading and vectorization. The reduction clause ensures
       each thread maintains a private copy of min_val initialized to
       DBL_MAX and the final result is the minimum across all threads.
    */
#pragma omp parallel for simd reduction(min:min_val) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double x = a[i];
        if (x < min_val) {
            min_val = x;
        }
    }
    result[0] = min_val;
}
