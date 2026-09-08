/*
 * Optimized implementation of the TSVC ``fuse_diamond`` kernel for double precision.
 *
 * The reference implementation computes a temporary array `t = a * a` and then multiplies
 * `(t + 1.0) * (t - 1.0)` into the output.  This version fuses the producer and the two
 * consumers into a single loop to reduce memory traffic and adds an OpenMP parallel+ SIMD
 * directive to exploit multi‑core and vector units.
 *
 * The function signature, argument types and naming follow the ABI expected by the benchmark
 * harness:
 *   void fuse_diamond_fp64(const double *restrict a,
 *                         double *restrict out,
 *                         const int64_t LEN_1D);
 */

#include <stdint.h>
#include <omp.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    /*
     * Parallelise over the outer dimension.  The loop body is free of loop‑carried
     * dependencies – each iteration reads a distinct element of `a` and writes a distinct
     * element of `out`.  Adding `simd` enables the compiler to emit vectorised code for the
     * inner iteration while `parallel for` distributes the work across threads.
     */
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double t = a[i] * a[i];
        out[i] = (t + 1.0) * (t - 1.0);
    }
}
