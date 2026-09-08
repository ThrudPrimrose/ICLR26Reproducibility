#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    int64_t n = LEN_1D - 1;
    if (n <= 0) return;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int T = omp_get_num_threads();

        int64_t chunk_size = (n + T - 1) / T;
        chunk_size = (chunk_size + 7) & ~7;

        int64_t start = (int64_t)tid * chunk_size;
        int64_t end = start + chunk_size;
        if (end > n) end = n;

        double boundary = 0.0;
        int is_last = 0;
        if (start < n) {
            is_last = (end == n);
            if (!is_last) {
                boundary = a[end];
            }
        }

        #pragma omp barrier

        if (start < n) {
            int64_t i = start;
            if (!is_last) {
                // Process vectors up to the last one with normal loads
                int64_t last_vec_start = end - 8;
                for (; i + 8 <= last_vec_start; i += 8) {
                    __m512d va = _mm512_loadu_pd(&a[i + 1]);
                    __m512d vb = _mm512_loadu_pd(&b[i]);
                    _mm512_storeu_pd(&a[i], _mm512_add_pd(va, vb));
                }
                // Last vector: lanes 0-6 from memory, lane 7 from boundary
                __m512d va_part = _mm512_maskz_loadu_pd(0x7F, &a[i + 1]);
                __m512d va_bound = _mm512_set1_pd(boundary);
                __m512d va = _mm512_mask_mov_pd(va_part, 0x80, va_bound);
                __m512d vb = _mm512_loadu_pd(&b[i]);
                _mm512_storeu_pd(&a[i], _mm512_add_pd(va, vb));
                i += 8;
            } else {
                for (; i + 8 <= end; i += 8) {
                    __m512d va = _mm512_loadu_pd(&a[i + 1]);
                    __m512d vb = _mm512_loadu_pd(&b[i]);
                    _mm512_storeu_pd(&a[i], _mm512_add_pd(va, vb));
                }
            }
            // Scalar tail (only for last thread if n not multiple of 8)
            for (; i < end; ++i) {
                a[i] = a[i + 1] + b[i];
            }
        }
    }
}
