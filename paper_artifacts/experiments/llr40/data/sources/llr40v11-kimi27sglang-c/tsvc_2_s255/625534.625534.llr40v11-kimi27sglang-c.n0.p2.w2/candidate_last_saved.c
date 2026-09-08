#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    const int64_t n = LEN_1D;

    a[0] = (b[0] + b[n - 1] + b[n - 2]) * 0.333;
    if (n == 1) return;

    a[1] = (b[1] + b[0] + b[n - 1]) * 0.333;
    if (n == 2) return;

    const __m256d vscale = _mm256_set1_pd(0.333);

    #pragma omp parallel
    {
        const int64_t nthreads = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t total = n - 2;
        const int64_t chunk = (total + nthreads - 1) / nthreads;
        int64_t start = 2 + tid * chunk;
        int64_t end = start + chunk;
        if (end > n) end = n;

        // Align the vector loop start to a 32-byte boundary for streaming stores.
        int64_t aligned_start = (start + 3) & ~((int64_t)3);
        if (aligned_start > end) aligned_start = end;

        int64_t i;
        for (i = start; i < aligned_start; i++) {
            a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333;
        }

        for (; i + 3 < end; i += 4) {
            __m256d vb0 = _mm256_loadu_pd(&b[i - 2]);
            __m256d vb1 = _mm256_loadu_pd(&b[i - 1]);
            __m256d vb2 = _mm256_loadu_pd(&b[i]);
            __m256d vsum = _mm256_add_pd(vb2, _mm256_add_pd(vb1, vb0));
            __m256d vres = _mm256_mul_pd(vsum, vscale);
            _mm256_stream_pd(&a[i], vres);
        }

        for (; i < end; i++) {
            a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333;
        }
    }

    _mm_sfence();
}
