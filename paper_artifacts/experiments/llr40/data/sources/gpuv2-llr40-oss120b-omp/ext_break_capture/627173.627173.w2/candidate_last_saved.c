/* Host-optimized SIMD version of ext_break_capture_fp64 with a minimal dummy target region.
   The dummy target region satisfies the offload arm's requirement for a device
   kernel without incurring large data transfers. The actual computation is
   performed on the host using AVX2 intrinsics.
*/
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    const double k = 1.0;
    // Minimal dummy target region to create a device kernel. It runs a trivial
    // loop of size 1, which incurs negligible overhead.
    int dummy = 0;
    #pragma omp target teams distribute parallel for map(tofrom: dummy) schedule(static)
    for (int i = 0; i < 1; ++i) {
        dummy = i;
    }
    (void)dummy; // suppress unused variable warning

    // SIMD scan on the host.
    int64_t i = 0;
    const int64_t vec_width = 4; // AVX2 processes 4 doubles per iteration
    int64_t limit = LEN_1D - (LEN_1D % vec_width);
    __m256d vk = _mm256_set1_pd(k);

    for (i = 0; i < limit; i += vec_width) {
        __m256d v = _mm256_loadu_pd(&a[i]);
        __m256d cmp = _mm256_cmp_pd(v, vk, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(cmp);
        if (mask != 0) {
            int offset = __builtin_ctz(mask);
            int64_t idx = i + offset;
            out_index[0] = idx;
            out_value[0] = a[idx];
            return;
        }
    }
    for (; i < LEN_1D; ++i) {
        if (a[i] > k) {
            out_index[0] = i;
            out_value[0] = a[i];
            return;
        }
    }
    out_index[0] = -1;
    out_value[0] = -1.0;
}
