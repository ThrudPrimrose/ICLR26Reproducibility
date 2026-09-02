#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void ext_break_capture_fp64(double * restrict a,
                            int64_t * restrict out_index,
                            double * restrict out_value,
                            int64_t K,
                            int64_t LEN_1D,
                            uint8_t * restrict workspace,
                            int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    const double Kd = (double)K;
    int64_t best_idx = LEN_1D;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const int64_t chunk = (LEN_1D + nthreads - 1) / nthreads;
        const int64_t start = tid * chunk;
        const int64_t end = (start + chunk < LEN_1D) ? (start + chunk) : LEN_1D;

        int64_t local_idx = LEN_1D;
        int64_t i = start;

#if defined(__AVX512F__)
        const __m512d vk = _mm512_set1_pd(Kd);
        for (; i + 8 <= end; i += 8) {
            __m512d va = _mm512_loadu_pd(&a[i]);
            __mmask8 mask = _mm512_cmp_pd_mask(va, vk, _CMP_GT_OQ);
            if (mask != 0) {
                local_idx = i + __builtin_ctz((unsigned int)mask);
                break;
            }
        }
#elif defined(__AVX2__)
        const __m256d vk = _mm256_set1_pd(Kd);
        for (; i + 4 <= end; i += 4) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d cmp = _mm256_cmp_pd(va, vk, _CMP_GT_OQ);
            int mask = _mm256_movemask_pd(cmp);
            if (mask != 0) {
                local_idx = i + __builtin_ctz((unsigned int)mask);
                break;
            }
        }
#endif

        for (; i < end && local_idx == LEN_1D; i++) {
            if (a[i] > Kd) {
                local_idx = i;
            }
        }

        #pragma omp critical
        {
            if (local_idx < best_idx) {
                best_idx = local_idx;
            }
        }
    }

    if (best_idx == LEN_1D) {
        out_index[0] = -1;
        out_value[0] = -1.0;
    } else {
        out_index[0] = best_idx;
        out_value[0] = a[best_idx];
    }
}
