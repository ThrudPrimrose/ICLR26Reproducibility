/* Optimized C implementation for segment_reduce_ragged kernel.
 * Performs a segmented dot product: for each segment s, compute sum_{e=row_ptr[s]}^{row_ptr[s+1]-1} val[e] * w[e]
 * and store the result in out[s].
 *
 * The outer loop over segments is independent, allowing parallel execution.
 * The inner loop is vectorizable; we use an OpenMP SIMD reduction to enable the compiler to emit SIMD code.
 *
 * The function follows the C v2 ABI expected by the benchmark harness.
 */

#include <stdint.h>
#include <stdio.h>

void segment_reduce_ragged_fp64(const int64_t *restrict row_ptr,
                               const double *restrict val,
                               const double *restrict w,
                               double *restrict out,
                               const int64_t NSEG) {
    printf("DEBUG: NSEG=%lld\n", (long long)NSEG);
    fflush(stdout);
    /* Parallelize the outer loop. The workload per segment varies, so a dynamic schedule helps balance threads.
     * Use a private accumulator per segment; the SIMD reduction ensures the inner loop vectorizes.
     */
    #pragma omp parallel for schedule(dynamic) shared(row_ptr, val, w, out, NSEG)
    for (int64_t s = 0; s < NSEG; ++s) {
        double acc = 0.0;
        const int64_t start = row_ptr[s];
        const int64_t end   = row_ptr[s + 1];
        /* The inner loop length is data-dependent; we still request vectorization.
         * The reduction clause combines the partial results produced by the SIMD lanes.
         */
        #pragma omp simd reduction(+:acc)
        for (int64_t e = start; e < end; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
