#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

/* TSVC tsvc_2 s255: a[i] = (b[i] + x + y) * 0.333 where x,y carry
 * b[(i-1) mod N], b[(i-2) mod N] unchanged -- a wrapped 3-point stencil
 * with no carry math, so the loop is fully parallel.
 *
 * The ops are written as explicit add/add/mul intrinsics so the result is
 * bit-identical to the serial reference ((b[i]+b[i-1])+b[i-2])*0.333:
 * no FMA contraction or reassociation is possible at this level. */
void tsvc_2_s255_fp64(double* a, double* b, int64_t LEN_1D, uint8_t* workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D <= 0) return;
    if (LEN_1D == 1) {
        a[0] = (b[0] + b[0] + b[0]) * 0.333;
        return;
    }
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333;
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333;

    const int64_t n = LEN_1D;
    const int64_t ve = 2 + ((n - 2) & ~(int64_t)7); /* 8-aligned vector end */
    const __m512d cc = _mm512_set1_pd(0.333);
    double* __restrict ar = a;
    const double* __restrict br = b;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 2; i < ve; i += 8) {
        const __m512d v0 = _mm512_loadu_pd(br + i);
        const __m512d v1 = _mm512_loadu_pd(br + i - 1);
        const __m512d v2 = _mm512_loadu_pd(br + i - 2);
        const __m512d acc = _mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(v0, v1), v2), cc);
        _mm_storeu_pd(ar + i + 0, _mm512_extractf64x2_pd(acc, 0));
        _mm_storeu_pd(ar + i + 2, _mm512_extractf64x2_pd(acc, 1));
        _mm_storeu_pd(ar + i + 4, _mm512_extractf64x2_pd(acc, 2));
        _mm_storeu_pd(ar + i + 6, _mm512_extractf64x2_pd(acc, 3));
    }
    for (int64_t i = ve; i < n; i++)
        ar[i] = (br[i] + br[i - 1] + br[i - 2]) * 0.333;
}
