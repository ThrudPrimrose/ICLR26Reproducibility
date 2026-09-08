/* Optimized version of tsvc_2_vag using OpenMP parallelism and AVX2 gather intrinsics.
   The kernel performs a gather: a[i] = b[ip[i]].
   For sufficiently large arrays we parallelize across threads and process 4 elements at a time
   using AVX2 256-bit vector gather. For small arrays we fall back to a scalar loop to avoid
   overhead.
*/

#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    const int64_t PARALLEL_THRESHOLD = 100000; // switch to parallel vectorized version for large inputs
    int64_t vec_end = LEN_1D - (LEN_1D % 4); // number of elements that can be processed in groups of 4

    if (LEN_1D >= PARALLEL_THRESHOLD) {
        // Parallel vectorized loop processing 4 doubles per iteration using AVX2 gather.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < vec_end; i += 4) {
            // Load four 32‑bit indices from ip.
            __m128i idx = _mm_loadu_si128((const __m128i *)(ip + i));
            // Gather four double values from b using the indices. Scale factor is sizeof(double) = 8.
            __m256d vals = _mm256_i32gather_pd(b, idx, 8);
            // Store the gathered values into a.
            _mm256_storeu_pd(a + i, vals);
        }
        // Process any remaining elements that don't fit into a group of four.
        for (int64_t i = vec_end; i < LEN_1D; ++i) {
            a[i] = b[(int64_t)ip[i]];
        }
    } else {
        // Small problem size – use a simple scalar loop.
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] = b[(int64_t)ip[i]];
        }
    }
}
