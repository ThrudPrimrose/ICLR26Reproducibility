#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void fuse_diamond_fp64(double * restrict a_arg, double * restrict out_arg, int64_t LEN_1D, uint8_t * restrict workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    #pragma omp parallel
    {
        int64_t num_threads = omp_get_num_threads();
        int64_t tid = omp_get_thread_num();
        int64_t chunk = (LEN_1D + num_threads - 1) / num_threads;
        int64_t start = tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;

        int64_t i = start;

        // Peel to align output pointer to 64 bytes
        while (i < end && (((uintptr_t)&out_arg[i]) & 63)) {
            double t = a_arg[i] * a_arg[i];
            out_arg[i] = (t + 1.0) * (t - 1.0);
            i++;
        }

        __m512d one = _mm512_set1_pd(1.0);

        for (; i + 8 <= end; i += 8) {
            __m512d a = _mm512_loadu_pd(&a_arg[i]);
            __m512d t = _mm512_mul_pd(a, a);
            __m512d u = _mm512_add_pd(t, one);
            __m512d v = _mm512_sub_pd(t, one);
            __m512d r = _mm512_mul_pd(u, v);
            _mm512_stream_pd(&out_arg[i], r);
        }

        for (; i < end; i++) {
            double t = a_arg[i] * a_arg[i];
            out_arg[i] = (t + 1.0) * (t - 1.0);
        }
    }

    _mm_sfence();
}
