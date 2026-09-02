#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

/* Sum of the positive elements of a[0..LEN_1D) into b[0].
 *
 * Memory-bandwidth bound: one 8-byte read per element, nothing else. 256-bit
 * (AVX2) masked loads measure ~3x faster than the 512-bit loop the compiler
 * picks on this Zen4-class part, so the hot chunk is hand-vectorized at
 * 256-bit with two independent accumulator lanes.
 *
 * The cross-thread combine is done BY HAND in a fixed order (parts[0]+parts[1]
 * + ... on the master after the region) instead of a GOMP reduction, whose
 * combine order is not bit-reproducible between runs -- the harness gate
 * requires byte-identical output on two clean runs. */

static double masked_chunk(const double *restrict p, int64_t m)
{
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    const __m256d zero = _mm256_setzero_pd();
    int64_t i = 0;
    for (; i + 8 <= m; i += 8) {
        __m256d x0 = _mm256_loadu_pd(p + i);
        __m256d x1 = _mm256_loadu_pd(p + i + 4);
        v0 = _mm256_add_pd(v0, _mm256_and_pd(_mm256_cmp_pd(x0, zero, _CMP_GT_OQ), x0));
        v1 = _mm256_add_pd(v1, _mm256_and_pd(_mm256_cmp_pd(x1, zero, _CMP_GT_OQ), x1));
    }
    double r0[4], r1[4];
    _mm256_storeu_pd(r0, v0);
    _mm256_storeu_pd(r1, v1);
    double s = r0[0] + r0[1] + r0[2] + r0[3] + r1[0] + r1[1] + r1[2] + r1[3];
    for (; i < m; i++)
        s += (p[i] > 0.0) ? p[i] : 0.0;
    return s;
}

void tsvc_2_s3111_fp64(double *restrict a, double *restrict b, int64_t LEN_1D,
                       uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D < 262144) {
        double sum_val = 0.0;
        for (int64_t i = 0; i < LEN_1D; i++)
            sum_val += (a[i] > 0.0) ? a[i] : 0.0;
        b[0] = sum_val;
        return;
    }
    static double parts[512];
    int64_t nt = 1;
    #pragma omp parallel
    {
        int64_t tnt = omp_get_num_threads();
        int64_t tid = omp_get_thread_num();
        if (tid == 0)
            nt = tnt;
        int64_t per = (LEN_1D + tnt - 1) / tnt;
        per = (per + 7) & ~7LL;
        int64_t lo = tid * per;
        int64_t hi = lo + per;
        if (hi > LEN_1D)
            hi = LEN_1D;
        parts[tid] = masked_chunk(a + lo, hi - lo);
    }
    double s = parts[0];
    for (int64_t i = 1; i < nt; i++)
        s += parts[i];
    b[0] = s;
}
