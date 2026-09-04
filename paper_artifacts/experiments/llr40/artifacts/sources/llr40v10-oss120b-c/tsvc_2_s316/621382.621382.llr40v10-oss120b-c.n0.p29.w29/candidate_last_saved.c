/*
 * Optimized implementation of the TSVC 2 "s316" kernel.
 *
 * This kernel computes the minimum of the input array `a` of length LEN_1D and
 * stores the result in `result[0]`.
 *
 * The original reference implementation uses a scalar loop starting from the
 * first element.  This version uses an OpenMP parallel reduction with the `min`
 * operator to exploit multi‑core CPUs, and the loop body is written in a way that
 * allows the compiler to auto‑vectorise the reduction.
 */

#include <stdint.h>
#include <float.h>
#include <math.h>
#include <omp.h>

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
    /* Use DBL_MAX as the identity for the min reduction.  For any finite input
     * value this is larger, so the reduction will correctly compute the global
     * minimum even when the loop iterates from i = 0.
     */
    double min_val = DBL_MAX;
    #pragma omp parallel for simd reduction(min:min_val) aligned(a:64)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        if (ai < min_val) {
            min_val = ai;
        }
    }
    result[0] = min_val;
}

