#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <stdint.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
    if (LEN_1D < 8192) {
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] += b[i] * S;
        }
        return;
    }

    const double s = (double)S;
#pragma omp parallel default(none) shared(a, b, LEN_1D) firstprivate(s)
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t base = LEN_1D / nt;
        const int64_t rem = LEN_1D % nt;
        const int64_t start = tid * base + (tid < rem ? tid : rem);
        const int64_t count = base + (tid < rem ? 1 : 0);

        double *restrict ap = a + start;
        const double *restrict bp = b + start;
        int64_t n = count;

        while (n > 0 && (((uintptr_t)ap) & 63)) {
            *ap += *bp * s;
            ++ap; ++bp; --n;
        }

        const int64_t vl = n & ~7;
        const __m512d sv = _mm512_set1_pd(s);
        for (int64_t i = 0; i < vl; i += 8) {
            __m512d av = _mm512_loadu_pd(ap + i);
            __m512d bv = _mm512_loadu_pd(bp + i);
            __m512d rv = _mm512_fmadd_pd(bv, sv, av);
            _mm512_stream_pd(ap + i, rv);
        }
        for (int64_t i = vl; i < n; ++i) {
            ap[i] += bp[i] * s;
        }
    }
    _mm_sfence();
}
