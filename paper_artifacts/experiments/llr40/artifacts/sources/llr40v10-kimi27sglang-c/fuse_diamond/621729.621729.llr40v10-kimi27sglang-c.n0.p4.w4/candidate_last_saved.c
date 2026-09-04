#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    if (LEN_1D < 4096) {
        const double *restrict pa = __builtin_assume_aligned(a, 64);
        double *restrict po = __builtin_assume_aligned(out, 64);
        #pragma omp parallel for simd schedule(static) aligned(pa, po : 64)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double t = pa[i] * pa[i];
            po[i] = (t + 1.0) * (t - 1.0);
        }
        return;
    }

    const __m512d one = _mm512_set1_pd(1.0);
    int64_t total_vec = LEN_1D >> 3;  // number of 8-element blocks
    int64_t rem_start = total_vec << 3;

    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t v_start = (int64_t)tid * total_vec / nt;
        int64_t v_end = (int64_t)(tid + 1) * total_vec / nt;
        int64_t i_start = v_start << 3;
        int64_t i_end = v_end << 3;
        for (int64_t i = i_start; i < i_end; i += 8) {
            __m512d va = _mm512_loadu_pd(&a[i]);
            __m512d vt = _mm512_mul_pd(va, va);
            __m512d vp = _mm512_add_pd(vt, one);
            __m512d vm = _mm512_sub_pd(vt, one);
            __m512d vo = _mm512_mul_pd(vp, vm);
            _mm512_stream_pd(&out[i], vo);
        }
    }
    _mm_sfence();

    for (int64_t i = rem_start; i < LEN_1D; ++i) {
        double t = a[i] * a[i];
        out[i] = (t + 1.0) * (t - 1.0);
    }
}
