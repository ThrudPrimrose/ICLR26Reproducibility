/* TSVC s2275:  aa[j,i] += bb[j,i]*cc[j,i]  (row-major -> flat contiguous FMA),
 *              a[i]      =  b[i] + c[i]*d[i]
 * The TSVC puzzle (distribute, then interchange) is solved by flattening: both
 * statements are element-wise over contiguous memory, so the kernel is a pure
 * streaming FMA: 3 reads + 1 write per element, AVX-512 vectorized, and
 * OpenMP-parallel over one NUMA-socket core set for large inputs.
 * Small inputs run serially: fork/join would cost more than the work. */
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>
#include <omp.h>

static __attribute__((always_inline)) inline void
fma_block(double *restrict aa, const double *restrict bb,
          const double *restrict cc, int64_t start, int64_t stop)
{
    int64_t k = start;
    const int64_t n32 = (stop - start) / 32;
    for (int64_t iv = 0; iv < n32; ++iv) {
        double *pa = aa + k;
        const double *pb = bb + k;
        const double *pc = cc + k;
        __m512d a0 = _mm512_loadu_pd(pa);
        __m512d b0 = _mm512_loadu_pd(pb);
        __m512d c0 = _mm512_loadu_pd(pc);
        __m512d a1 = _mm512_loadu_pd(pa + 8);
        __m512d b1 = _mm512_loadu_pd(pb + 8);
        __m512d c1 = _mm512_loadu_pd(pc + 8);
        __m512d a2 = _mm512_loadu_pd(pa + 16);
        __m512d b2 = _mm512_loadu_pd(pb + 16);
        __m512d c2 = _mm512_loadu_pd(pc + 16);
        __m512d a3 = _mm512_loadu_pd(pa + 24);
        __m512d b3 = _mm512_loadu_pd(pb + 24);
        __m512d c3 = _mm512_loadu_pd(pc + 24);
        _mm512_storeu_pd(pa,      _mm512_fmadd_pd(b0, c0, a0));
        _mm512_storeu_pd(pa + 8,  _mm512_fmadd_pd(b1, c1, a1));
        _mm512_storeu_pd(pa + 16, _mm512_fmadd_pd(b2, c2, a2));
        _mm512_storeu_pd(pa + 24, _mm512_fmadd_pd(b3, c3, a3));
        k += 32;
    }
    for (; k < stop; ++k)
        aa[k] = aa[k] + bb[k] * cc[k];
}

void tsvc_2_s2275_fp64(double *restrict a,
                       double *restrict aa,
                       const double *restrict b,
                       const double *restrict bb,
                       const double *restrict c,
                       const double *restrict cc,
                       const double *restrict d,
                       const int64_t LEN_2D,
                       uint8_t *restrict workspace,
                       const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    const int64_t N = LEN_2D;
    const int64_t total = N * N;

    if (total >= 1 << 16) {
        #pragma omp parallel
        {
            const int64_t nt = omp_get_num_threads();
            const int64_t tid = omp_get_thread_num();
            fma_block(aa, bb, cc, (total * tid) / nt, (total * (tid + 1)) / nt);
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < N; ++i)
                a[i] = b[i] + c[i] * d[i];
        }
    } else {
        fma_block(aa, bb, cc, 0, total);
        for (int64_t i = 0; i < N; ++i)
            a[i] = b[i] + c[i] * d[i];
    }
}
