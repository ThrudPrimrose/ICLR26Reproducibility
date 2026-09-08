#include <stdint.h>
#include <immintrin.h>

static inline __m512d prefix_sum_8(__m512d x) {
    const __m512i idx1 = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 0);
    const __m512i idx2 = _mm512_set_epi64(5, 4, 3, 2, 1, 0, 0, 0);
    const __m512i idx3 = _mm512_set_epi64(3, 2, 1, 0, 0, 0, 0, 0);
    x = _mm512_add_pd(x, _mm512_maskz_permutexvar_pd(0xFE, idx1, x));
    x = _mm512_add_pd(x, _mm512_maskz_permutexvar_pd(0xFC, idx2, x));
    x = _mm512_add_pd(x, _mm512_maskz_permutexvar_pd(0xF0, idx3, x));
    return x;
}

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    int64_t i = 0;

    /* scalar head until b is aligned for streaming stores */
    while (i < LEN_1D && (((uintptr_t)(b + i)) & 63)) {
        sum += a[i];
        b[i] = sum;
        ++i;
    }

    const int64_t n_vec = ((LEN_1D - i) >> 3) << 3;
    const int64_t end_vec = i + n_vec;
    const int64_t end_unroll = i + ((n_vec >> 5) << 5);

    __m512d vsum = _mm512_set1_pd(sum);

    for (; i < end_unroll; i += 32) {
        __m512d x0 = _mm512_loadu_pd(a + i + 0);
        __m512d x1 = _mm512_loadu_pd(a + i + 8);
        __m512d x2 = _mm512_loadu_pd(a + i + 16);
        __m512d x3 = _mm512_loadu_pd(a + i + 24);

        double t0 = _mm512_reduce_add_pd(x0);
        double t1 = _mm512_reduce_add_pd(x1);
        double t2 = _mm512_reduce_add_pd(x2);
        double t3 = _mm512_reduce_add_pd(x3);

        __m512d p0 = prefix_sum_8(x0);
        __m512d p1 = prefix_sum_8(x1);
        __m512d p2 = prefix_sum_8(x2);
        __m512d p3 = prefix_sum_8(x3);

        __m512d off0 = vsum;
        __m512d off1 = _mm512_add_pd(off0, _mm512_set1_pd(t0));
        __m512d off2 = _mm512_add_pd(off1, _mm512_set1_pd(t1));
        __m512d off3 = _mm512_add_pd(off2, _mm512_set1_pd(t2));

        _mm512_stream_pd(b + i + 0, _mm512_add_pd(p0, off0));
        _mm512_stream_pd(b + i + 8, _mm512_add_pd(p1, off1));
        _mm512_stream_pd(b + i + 16, _mm512_add_pd(p2, off2));
        _mm512_stream_pd(b + i + 24, _mm512_add_pd(p3, off3));

        vsum = _mm512_add_pd(off3, _mm512_set1_pd(t3));
    }

    _mm_sfence();

    /* remaining full vectors (less than 4) */
    for (; i < end_vec; i += 8) {
        __m512d x = _mm512_loadu_pd(a + i);
        __m512d p = prefix_sum_8(x);
        __m512d y = _mm512_add_pd(p, vsum);
        _mm512_stream_pd(b + i, y);
        vsum = _mm512_set1_pd(_mm512_mask_reduce_add_pd(0x80, y));
    }

    _mm_sfence();

    sum = _mm_cvtsd_f64(_mm512_castpd512_pd128(vsum));

    for (; i < LEN_1D; ++i) {
        sum += a[i];
        b[i] = sum;
    }
}
