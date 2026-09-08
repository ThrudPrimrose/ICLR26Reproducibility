#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>

/*
 * High‑performance row‑major implementation of the TSVC tsvc_2 ``s235`` kernel.
 *
 * The algorithm updates the ``a`` vector (embarrassingly parallel) and then
 * computes a column‑wise prefix sum using a per‑column accumulator that lives in
 * a heap‑allocated array.  The inner loop over columns is vectorised with
 * ``#pragma omp simd``; the accumulator array is accessed contiguously, allowing
 * the compiler to emit efficient SIMD code.  This formulation minimises memory
 * traffic: ``bb`` and ``aa`` are streamed once each (row‑major order) and the
 * additional reads of ``a`` are cache‑friendly.
 */

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    /* 1. Update the ``a`` vector – independent work, parallelised with OpenMP. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] += b[i] * c[i];
    }

    /* 2. Allocate a per‑column accumulator.  ``malloc`` ensures the buffer is
       suitably aligned for SIMD loads/stores on typical systems. */
    double *restrict acc = (double *)malloc((size_t)LEN_2D * sizeof(double));
    if (!acc) return;  // allocation failure – nothing to do.

    /* Initialise the accumulator from the first row of ``aa``. */
    for (int64_t i = 0; i < LEN_2D; ++i) {
        acc[i] = aa[i];
    }

    /* 3. Process the remaining rows sequentially.  The inner ``i`` loop touches
       data in contiguous order, which enables the compiler to generate a
       vectorised implementation (SIMD width up to 4‑wide double on AVX2).
       ``#pragma omp simd`` forces vectorisation even in the presence of the
       accumulator dependency. */
    for (int64_t j = 1; j < LEN_2D; ++j) {
        int64_t row_offset = j * LEN_2D;
        #pragma omp simd
        for (int64_t i = 0; i < LEN_2D; ++i) {
            double val = acc[i] + bb[row_offset + i] * a[i];
            acc[i] = val;
            aa[row_offset + i] = val;
        }
    }

    free(acc);
}
