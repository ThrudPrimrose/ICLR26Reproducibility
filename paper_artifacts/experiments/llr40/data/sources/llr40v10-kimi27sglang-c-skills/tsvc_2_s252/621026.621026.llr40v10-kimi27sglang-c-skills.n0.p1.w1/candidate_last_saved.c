#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

// Per-thread worker: a[i] = b[i]*c[i] + (i==start ? prev : b[i-1]*c[i-1]).
// The output stream is aligned to 64 bytes via a scalar prefix before vectorising.
static inline void scan_chunk(double *restrict a,
                              const double *restrict b,
                              const double *restrict c,
                              int64_t i, int64_t end, double prev) {
    for (; i < end && ((uintptr_t)&a[i] & 63) != 0; ++i) {
        double s = b[i] * c[i];
        a[i] = s + prev;
        prev = s;
    }

    __m512d vprev = _mm512_set1_pd(prev);
    __m512i lane7 = _mm512_set1_epi64(7);

    for (; i + 8 <= end; i += 8) {
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vc = _mm512_loadu_pd(&c[i]);
        __m512d s  = _mm512_mul_pd(vb, vc);

        // rotated = [s7, s0, s1, s2, s3, s4, s5, s6]
        __m512d rotated = _mm512_castsi512_pd(_mm512_alignr_epi64(
            _mm512_castpd_si512(s), _mm512_castpd_si512(s), 7));
        // shifted = [prev, s0, s1, s2, s3, s4, s5, s6]
        __m512d shifted = _mm512_mask_blend_pd(0x01, rotated, vprev);

        __m512d va = _mm512_add_pd(s, shifted);
        _mm512_stream_pd(&a[i], va);

        // carry s7 into the next block
        vprev = _mm512_permutexvar_pd(lane7, s);
    }

    prev = _mm512_cvtsd_f64(vprev);
    for (; i < end; ++i) {
        double s = b[i] * c[i];
        a[i] = s + prev;
        prev = s;
    }
}

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    const int64_t nthreads = omp_get_max_threads();
    const int64_t chunk = (LEN_1D + nthreads - 1) / nthreads;

    // For small inputs the OpenMP launch cost dominates; run serially.
    if (chunk < 8192) {
        scan_chunk(a, b, c, 0, LEN_1D, 0.0);
        _mm_sfence();
        return;
    }

    #pragma omp parallel
    {
        const int64_t tid = omp_get_thread_num();
        int64_t i = tid * chunk;
        int64_t end = i + chunk;
        if (end > LEN_1D) end = LEN_1D;

        double prev = 0.0;
        if (i > 0 && i < LEN_1D) {
            prev = b[i - 1] * c[i - 1];
        }

        scan_chunk(a, b, c, i, end, prev);
    }

    _mm_sfence();
}
