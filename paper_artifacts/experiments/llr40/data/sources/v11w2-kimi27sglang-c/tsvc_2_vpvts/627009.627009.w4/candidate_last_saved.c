#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
    const double s = (double)S;
    const int64_t n = LEN_1D;

    if (n <= 1024) {
        #pragma omp simd safelen(8)
        for (int64_t i = 0; i < n; ++i) {
            a[i] += b[i] * s;
        }
        return;
    }

    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        int64_t block = (n + nt - 1) / nt;
        block = (block + 7) & ~((int64_t)7);

        int64_t i0 = tid * block;
        if (i0 < n) {
            int64_t i1 = i0 + block;
            if (i1 > n) i1 = n;

            __m512d vs = _mm512_set1_pd(s);
            int64_t i = i0;

            while (i < i1 && ((uintptr_t)(a + i) & 63) != 0) {
                a[i] += b[i] * s;
                ++i;
            }

            for (; i + 32 <= i1; i += 32) {
                _mm_prefetch((const char *)(b + i + 64), _MM_HINT_T0);
                _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);

                __m512d vb0 = _mm512_loadu_pd(b + i);
                __m512d vb1 = _mm512_loadu_pd(b + i + 8);
                __m512d vb2 = _mm512_loadu_pd(b + i + 16);
                __m512d vb3 = _mm512_loadu_pd(b + i + 24);

                __m512d va0 = _mm512_load_pd(a + i);
                __m512d va1 = _mm512_load_pd(a + i + 8);
                __m512d va2 = _mm512_load_pd(a + i + 16);
                __m512d va3 = _mm512_load_pd(a + i + 24);

                va0 = _mm512_fmadd_pd(vb0, vs, va0);
                va1 = _mm512_fmadd_pd(vb1, vs, va1);
                va2 = _mm512_fmadd_pd(vb2, vs, va2);
                va3 = _mm512_fmadd_pd(vb3, vs, va3);

                _mm512_store_pd(a + i, va0);
                _mm512_store_pd(a + i + 8, va1);
                _mm512_store_pd(a + i + 16, va2);
                _mm512_store_pd(a + i + 24, va3);
            }

            for (; i + 8 <= i1; i += 8) {
                __m512d vb = _mm512_loadu_pd(b + i);
                __m512d va = _mm512_load_pd(a + i);
                va = _mm512_fmadd_pd(vb, vs, va);
                _mm512_store_pd(a + i, va);
            }

            for (; i < i1; ++i) {
                a[i] += b[i] * s;
            }
        }
    }
}
