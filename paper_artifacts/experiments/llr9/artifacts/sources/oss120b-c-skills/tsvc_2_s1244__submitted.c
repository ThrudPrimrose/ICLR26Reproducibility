/*
 * TSVC_2 kernel "s1244" implementation.
 *
 * Reference implementation (Python/NumPy) from the benchmark suite:
 *   a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
 *   d[i] = a[i] + a[i + 1]
 * for i in [0, LEN_1D-2].
 *
 * This C version follows the required signature, uses OpenMP for parallelism
 * and favours vectorisation. All pointer arguments are declared `restrict`
 * because the benchmark guarantees they do not alias. The length argument is
 * `int64_t` as required by the harness.
 */

#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

/*
 * Compute the s1244 kernel.
 *
 * Parameters:
 *   a      - output array, also used as intermediate storage for the first loop.
 *   b      - input array.
 *   c      - input array.
 *   d      - output array.
 *   LEN_1D - number of elements in each 1‑D array.
 *
 * The kernel assumes the caller has allocated `a`, `b`, `c`, `d` with at least
 * LEN_1D elements. The element a[LEN_1D‑1] is never written by the kernel – it
 * may contain any value because only a[i] for i < LEN_1D‑1 participates in the
 * computation of d[i].
 */
void tsvc_2_s1244_fp64(double *restrict a,
                       const double *restrict b,
                       const double *restrict c,
                       double *restrict d,
                       int64_t LEN_1D,
                       uint8_t *restrict workspace,
                       int64_t workspace_bytes) {
    double *a_orig = (double *)workspace;
    if (a_orig == NULL) {
        a_orig = (double *)malloc(sizeof(double) * (size_t)LEN_1D);
    }
    // copy original a into a_orig
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a_orig[i] = a[i];
    }
    const int64_t limit = LEN_1D - 1;
    (void)workspace_bytes;
    // Will free a_orig after second loop if allocated
    /* First loop: compute a[i] for i = 0 … LEN_1D‑2.
     * Independent across iterations – safe to parallelise and vectorise.
     */
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < limit; ++i) {
        double bi = b[i];
        double ci = c[i];
        a[i] = bi + ci * ci + bi * bi + ci;
    }

    /* Second loop: compute d[i] = a[i] + a[i+1] for the same range.
     * All reads are from the already‑filled `a` array, therefore the loop is
     * embarrassingly parallel.
     */
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < limit; ++i) {
        d[i] = a[i] + a_orig[i + 1];
    }
    if (workspace == NULL) {
        free(a_orig);
    }
}
