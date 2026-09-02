#include <stdint.h>
#include <stddef.h>
#include <omp.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

/*
 * TSVC vag:  for i in 0..LEN_1D-1:  a[i] = b[ip[i]]   (ip is int32)
 *
 * Pure gather: a[i] written once, never read; ip, b read-only.  No
 * dependences on any axis -> fully parallel; static schedule gives each
 * thread a wide contiguous span.
 *
 * AVX-512: 16 elements per step -- two independent (load, gather, stream)
 * triples back-to-back for extra memory-level parallelism per thread;
 * non-temporal stores because a is a write-once streaming array.  Head
 * peeled to the 64B boundary the streaming store needs (a is 8B-aligned
 * from the ABI), at most 7 head + 15 tail elements scalar.
 */
void tsvc_2_vag_fp64(double *restrict a, const double *restrict b,
                     const int32_t *restrict ip, int64_t LEN_1D,
                     uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

#ifdef __AVX512F__
    {
        const int64_t n16 = LEN_1D - (LEN_1D & 15);
        int64_t i0 = (64 - ((uintptr_t)a & 63)) & 63;
        i0 = (i0 + 15) & ~(int64_t)15;
        if (i0 > n16)
            i0 = n16;

        #pragma omp parallel for schedule(static)
        for (int64_t i = i0; i < n16; i += 16) {
            __m256i i1 = _mm256_loadu_si256((const __m256i *)(ip + i));
            __m256i i2 = _mm256_loadu_si256((const __m256i *)(ip + i + 8));
            __m512d v1 = _mm512_i32gather_pd(i1, b, 8);
            __m512d v2 = _mm512_i32gather_pd(i2, b, 8);
            _mm512_stream_pd(a + i, v1);
            _mm512_stream_pd(a + i + 8, v2);
        }
        for (int64_t i = 0; i < i0; i++)
            a[i] = b[ip[i]];
        for (int64_t i = n16; i < LEN_1D; i++)
            a[i] = b[ip[i]];
        return;
    }
#endif

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; i++)
        a[i] = b[ip[i]];
}
