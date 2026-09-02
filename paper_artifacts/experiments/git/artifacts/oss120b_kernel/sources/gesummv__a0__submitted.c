// Generated optimized version of gesummv kernel
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

/* Portable complex conjugate helper (kept from reference) */
static inline double _Complex __npb_conj(double _Complex z) {
    return __builtin_complex(__real__ z, -__imag__ z);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#ifdef I
#undef I
#endif

/* Minimal version of min/max macros (not used in this kernel) */
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

/* Optimized gesummv kernel: performs out = alpha * A * x + beta * B * x
   where A, B are N x N matrices stored in row-major order.
   Uses OpenMP parallelism and SIMD vectorization.
*/
void gesummv_fp64(const double *restrict A, const double *restrict B,
                   double *restrict out, const double *restrict x,
                   int64_t N, double alpha, double beta) {
    // Parallelize outer loop over rows
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        double sum_a = 0.0;
        double sum_b = 0.0;
        const double *a_row = A + i * N;
        const double *b_row = B + i * N;
        // SIMD vectorize inner dot product, with reductions on the sums
        #pragma omp simd reduction(+:sum_a,sum_b)
        for (int64_t j = 0; j < N; ++j) {
            double xv = x[j];
            sum_a += a_row[j] * xv;
            sum_b += b_row[j] * xv;
        }
        out[i] = alpha * sum_a + beta * sum_b;
    }
}
