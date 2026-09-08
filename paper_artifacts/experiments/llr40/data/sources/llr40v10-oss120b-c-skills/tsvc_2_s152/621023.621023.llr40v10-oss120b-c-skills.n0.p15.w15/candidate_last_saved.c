/* Hand port of the TSVC tsvc_2 C++ microkernel ``s152`` (s152_d_single.cpp), fp64
 * single-invocation variant, to C23 under the v2 C-ABI.
 *
 * Adapted from TSVC_2 -- Test Suite for Vectorizing Compilers (github.com/UoB-HPC/TSVC_2),
 * NCSA/MIT license (UIUC).
 *
 * This implementation adds OpenMP parallelization and vectorization.
 */

#include <stdint.h>
#include <omp.h>

/* Helper kernel to perform the a[i] update.
 * Declared static inline so the compiler can inline it.
 */
static inline void s152s_kernel(double *restrict a, const double *restrict b, const double *restrict c, const int64_t i) {
    a[i] += b[i] * c[i];
}

/* Compute b[i] = d[i] * e[i]; then a[i] += b[i] * c[i] for each i.
 * All pointers are restrict-qualified as required by the ABI, ensuring no aliasing.
 */
void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e,
                      const int64_t LEN_1D) {
    /* Parallelize the outer loop across threads and enable SIMD vectorization.
     * The loop body has no cross‑iteration dependencies, so this is safe.
     */
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        b[i] = d[i] * e[i];
        s152s_kernel(a, b, c, i);
    }
}

