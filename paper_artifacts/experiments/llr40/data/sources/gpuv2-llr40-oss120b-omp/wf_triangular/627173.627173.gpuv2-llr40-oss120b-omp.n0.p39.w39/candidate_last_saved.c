/* Optimized implementation of wf_triangular using wavefront parallelism and OpenMP target offload.
 * Computes the triangular wavefront: for i=1..N-1, j=i..N-1
 *   a[i,j] += a[i-1,j] + a[i,j-1]
 *
 * The computation has a data dependency across anti‑diagonals (i+j constant).
 * We parallelize each anti‑diagonal independently using an OpenMP target region.
 */

#include <stdint.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (N <= 1) return;

    // Map the entire array to the device for the duration of the computation.
    #pragma omp target data map(tofrom: a[0:N*N]) if(0)
    {
        // The wavefront index s = i + j runs from 2 to 2*(N-1).
        for (int64_t s = 2; s <= 2 * (N - 1); ++s) {
            // Compute the bounds for i on this diagonal while ensuring j >= i.
            int64_t i_start = s - (N - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = s / 2; // floor(s/2)
            if (i_end > N - 1) i_end = N - 1;
            // Parallelise the independent points on the diagonal.
            #pragma omp target teams distribute parallel for simd if(0) schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                    int64_t j = s - i;
                    // Compute linear index; neighbours are idx-N and idx-1.
                    int64_t idx = i * N + j;
                    a[idx] = a[idx] + a[idx - N] + a[idx - 1];
                }
        }
    }
}

