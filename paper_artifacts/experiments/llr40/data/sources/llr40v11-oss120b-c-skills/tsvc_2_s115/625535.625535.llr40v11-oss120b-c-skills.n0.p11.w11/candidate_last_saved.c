#include <stddef.h>
#include <stdint.h>
#include <omp.h>

// Parallelized and vectorized version of tsvc_2_s115
// Computes a[i] -= aa[j * LEN_2D + i] * a[j] for i>j.
// The outer loop is sequential due to true dependence on a[j];
// the inner loop is parallelized across threads and vectorized.
void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
    #pragma omp parallel
    {
        for (int64_t j = 0; j < LEN_2D; ++j) {
            double aj = a[j]; // load once per outer iteration
            #pragma omp for simd schedule(static)
            for (int64_t i = j + 1; i < LEN_2D; ++i) {
                a[i] -= aa[j * LEN_2D + i] * aj;
            }
            // implicit barrier ensures all updates to a[i] complete before next j
        }
    }
}
