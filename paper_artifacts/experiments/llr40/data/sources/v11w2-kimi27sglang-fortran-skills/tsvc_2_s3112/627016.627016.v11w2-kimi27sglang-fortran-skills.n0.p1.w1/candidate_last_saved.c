#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    int64_t n = LEN_1D;
    if (n <= 0) return;

    int nt_env = omp_get_max_threads();
    int nt = nt_env;
    if (nt > 32) nt = 32;

    if (n < 4096 || nt <= 1) {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            s += a[i];
            b[i] = s;
        }
        return;
    }

    static double part[64];

    #pragma omp parallel num_threads(nt)
    {
        int t = omp_get_thread_num();
        int64_t lo = (n * (int64_t)t) / (int64_t)nt;
        int64_t hi = (n * (int64_t)(t + 1)) / (int64_t)nt;

        double run = 0.0;
        int64_t i = lo;
        if (hi - lo >= 8) {
            __m512d vsum = _mm512_setzero_pd();
            for (; i <= hi - 8; i += 8) {
                vsum = _mm512_add_pd(vsum, _mm512_loadu_pd(&a[i]));
            }
            run = _mm512_reduce_add_pd(vsum);
        }
        for (; i < hi; ++i) run += a[i];
        part[t + 1] = run;

        #pragma omp barrier
        #pragma omp single
        {
            for (int j = 1; j <= nt; ++j) {
                part[j] += part[j - 1];
            }
        }

        double offset = part[t];
        i = lo;

        // Peel to align b to 64 bytes for streaming stores
        uintptr_t baddr = (uintptr_t)&b[i];
        if ((baddr & 63) != 0) {
            int64_t peel = (64 - (baddr & 63)) / 8;
            int64_t align_end = i + peel;
            if (align_end > hi) align_end = hi;
            for (; i < align_end; ++i) {
                offset += a[i];
                b[i] = offset;
            }
        }

        const __m512i idx1 = _mm512_set_epi64(6,5,4,3,2,1,0,0);
        const __m512i idx2 = _mm512_set_epi64(5,4,3,2,1,0,0,0);
        const __m512i idx4 = _mm512_set_epi64(3,2,1,0,0,0,0,0);
        for (; i <= hi - 8; i += 8) {
            __m512d x = _mm512_loadu_pd(&a[i]);
            __m512d s = x;
            __m512d t1 = _mm512_maskz_permutexvar_pd(0xFE, idx1, s);
            s = _mm512_add_pd(s, t1);
            __m512d t2 = _mm512_maskz_permutexvar_pd(0xFC, idx2, s);
            s = _mm512_add_pd(s, t2);
            __m512d t4 = _mm512_maskz_permutexvar_pd(0xF0, idx4, s);
            s = _mm512_add_pd(s, t4);
            s = _mm512_add_pd(s, _mm512_set1_pd(offset));
            _mm512_stream_pd(&b[i], s);
            offset += _mm512_reduce_add_pd(x);
        }
        for (; i < hi; ++i) {
            offset += a[i];
            b[i] = offset;
        }
    }
    _mm_sfence();
}
